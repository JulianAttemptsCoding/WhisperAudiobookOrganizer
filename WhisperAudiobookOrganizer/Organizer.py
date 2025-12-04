#!/usr/bin/env python3
"""
Robust Whisper-based transcription and chapter-aware file renaming tool.
Processes MP3 files in order, transcribes them, identifies chapters,
and renames files according to the specified naming convention.
"""

import os
import re
import sys
import json
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ChapterInfo:
    """Data class to hold chapter information"""
    chapter_number: int
    track_number: int
    start_file_index: int
    start_time: Optional[float] = None

@dataclass
class TranscriptionResult:
    """Data class to hold transcription results"""
    filename: str
    index: int
    transcription: str
    success: bool
    error: Optional[str] = None

class WhisperProcessor:
    """Handles Whisper transcription operations"""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            import whisper
            self.whisper = whisper
        except ImportError:
            logger.error("OpenAI Whisper not found. Install with: pip install openai-whisper")
            raise ImportError("OpenAI Whisper is required")
        
        try:
            import torch
        except ImportError:
            logger.error("PyTorch not found. Install with: pip install torch")
            raise ImportError("PyTorch is required")
    
    def load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper {self.model_size} model...")
            self.model = self.whisper.load_model(self.model_size)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_file(self, file_path: Path) -> str:
        """Transcribe a single audio file"""
        try:
            result = self.model.transcribe(str(file_path), fp16=False, language="en")
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription failed for {file_path}: {e}")
            raise

class AudioFileProcessor:
    """Main processor for handling audio files and chapters"""
    
    def __init__(self, input_directory: Path, output_directory: Path, 
                 model_size: str = "base", max_workers: int = 4):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.max_workers = max_workers
        self.whisper_processor = WhisperProcessor(model_size)
        self.transcription_cache = {}
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        (self.output_directory / "transcriptions").mkdir(exist_ok=True)
        (self.output_directory / "originals").mkdir(exist_ok=True)
    
    def get_sorted_mp3_files(self) -> List[Path]:
        """Get all MP3 files sorted by name"""
        mp3_files = list(self.input_directory.glob("*.mp3"))
        mp3_files.sort(key=lambda x: x.name.lower())
        logger.info(f"Found {len(mp3_files)} MP3 files")
        return mp3_files
    
    def transcribe_files(self, files: List[Path]) -> List[TranscriptionResult]:
        """Transcribe all files with progress tracking"""
        results = []
        self.whisper_processor.load_model()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all transcription tasks
            future_to_file = {
                executor.submit(self._transcribe_single_file, file_path, idx): (file_path, idx)
                for idx, file_path in enumerate(files)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path, idx = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.success:
                        logger.info(f"Transcribed {idx + 1}/{len(files)}: {file_path.name}")
                    else:
                        logger.error(f"Failed {idx + 1}/{len(files)}: {file_path.name} - {result.error}")
                except Exception as e:
                    logger.error(f"Exception processing {file_path}: {e}")
                    results.append(TranscriptionResult(
                        filename=file_path.name,
                        index=idx,
                        transcription="",
                        success=False,
                        error=str(e)
                    ))
        
        # Sort results by original file order
        results.sort(key=lambda x: x.index)
        return results
    
    def _transcribe_single_file(self, file_path: Path, index: int) -> TranscriptionResult:
        """Transcribe a single file with error handling"""
        try:
            transcription = self.whisper_processor.transcribe_file(file_path)
            
            # Save transcription to file
            transcription_file = self.output_directory / "transcriptions" / f"{file_path.stem}.txt"
            transcription_file.write_text(transcription, encoding='utf-8')
            
            return TranscriptionResult(
                filename=file_path.name,
                index=index,
                transcription=transcription,
                success=True
            )
        except Exception as e:
            return TranscriptionResult(
                filename=file_path.name,
                index=index,
                transcription="",
                success=False,
                error=str(e)
            )
    
    def find_chapters(self, results: List[TranscriptionResult]) -> List[ChapterInfo]:
        """Find chapter markers in transcriptions"""
        chapters = []
        current_chapter = None
        track_counter = 1
        
        # Pattern to match "chapter [xx]" or "chapter xx" (case insensitive)
        chapter_pattern = re.compile(r'chapter\s*(\d+)', re.IGNORECASE)
        
        for idx, result in enumerate(results):
            if not result.success:
                continue
                
            # Find all chapter mentions in the transcription
            matches = chapter_pattern.findall(result.transcription)
            
            for match in matches:
                chapter_num = int(match)
                
                # If this is a new chapter or continuation of current chapter
                if current_chapter is None or current_chapter.chapter_number != chapter_num:
                    # Save previous chapter if it exists
                    if current_chapter is not None:
                        chapters.append(current_chapter)
                    
                    # Start new chapter
                    current_chapter = ChapterInfo(
                        chapter_number=chapter_num,
                        track_number=1,
                        start_file_index=idx
                    )
                    track_counter = 1
                    logger.info(f"Found Chapter {chapter_num} starting at file {idx + 1}: {result.filename}")
                else:
                    # Same chapter, increment track counter
                    track_counter += 1
                    if current_chapter:
                        current_chapter.track_number = track_counter
        
        # Don't forget the last chapter
        if current_chapter is not None:
            chapters.append(current_chapter)
        
        # Sort chapters by chapter number and then by file index
        chapters.sort(key=lambda x: (x.chapter_number, x.start_file_index))
        
        logger.info(f"Found {len(chapters)} chapter markers")
        for chapter in chapters:
            logger.info(f"  Chapter {chapter.chapter_number}, Track {chapter.track_number}, File {chapter.start_file_index + 1}")
        
        return chapters
    
    def rename_files(self, files: List[Path], results: List[TranscriptionResult], 
                    chapters: List[ChapterInfo]) -> None:
        """Rename files according to chapter and track numbers"""
        
        if not chapters:
            logger.warning("No chapters found, skipping renaming")
            return
        
        # Create chapter boundaries
        chapter_boundaries = []
        for i, chapter in enumerate(chapters):
            next_start = chapters[i + 1].start_file_index if i + 1 < len(chapters) else len(files)
            chapter_boundaries.append({
                'chapter': chapter,
                'end_index': next_start - 1
            })
        
        # Process each chapter
        for boundary in chapter_boundaries:
            chapter = boundary['chapter']
            end_index = boundary['end_index']
            
            # Process files in this chapter range
            track_number = 1
            for file_index in range(chapter.start_file_index, end_index + 1):
                if file_index >= len(files):
                    break
                    
                original_file = files[file_index]
                new_filename = f"{chapter.chapter_number}.{track_number}.EOE.mp3"
                new_filepath = self.output_directory / new_filename
                
                try:
                    # Copy file to output directory with new name
                    shutil.copy2(original_file, new_filepath)
                    logger.info(f"Renamed: {original_file.name} -> {new_filename}")
                    track_number += 1
                except Exception as e:
                    logger.error(f"Failed to copy {original_file} to {new_filename}: {e}")
        
        # Handle files before first chapter (if any)
        if chapters and chapters[0].start_file_index > 0:
            logger.info("Files before first chapter will be named as Chapter 0")
            for i in range(chapters[0].start_file_index):
                original_file = files[i]
                new_filename = f"0.{i + 1}.EOE.mp3"
                new_filepath = self.output_directory / new_filename
                try:
                    shutil.copy2(original_file, new_filepath)
                    logger.info(f"Renamed (pre-chapter): {original_file.name} -> {new_filename}")
                except Exception as e:
                    logger.error(f"Failed to copy {original_file} to {new_filename}: {e}")
    
    def process(self) -> None:
        """Main processing pipeline"""
        try:
            logger.info("Starting audio file processing pipeline")
            
            # Get sorted MP3 files
            files = self.get_sorted_mp3_files()
            if not files:
                logger.error("No MP3 files found in the input directory")
                return
            
            # Transcribe all files
            logger.info("Starting transcription process...")
            results = self.transcribe_files(files)
            
            # Save results summary
            summary_file = self.output_directory / "processing_summary.json"
            summary_data = {
                'total_files': len(files),
                'successful_transcriptions': len([r for r in results if r.success]),
                'failed_transcriptions': len([r for r in results if not r.success]),
                'results': [
                    {
                        'filename': r.filename,
                        'success': r.success,
                        'error': r.error
                    } for r in results
                ]
            }
            summary_file.write_text(json.dumps(summary_data, indent=2))
            
            # Find chapters
            chapters = self.find_chapters(results)
            
            # Rename files
            self.rename_files(files, results, chapters)
            
            logger.info("Processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Transcribe MP3 files and rename by chapters")
    parser.add_argument("input_directory", help="Directory containing MP3 files")
    parser.add_argument("output_directory", help="Directory for output files")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    # Validate directories
    input_dir = Path(args.input_directory)
    output_dir = Path(args.output_directory)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    try:
        processor = AudioFileProcessor(input_dir, output_dir, args.model, args.workers)
        processor.process()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()