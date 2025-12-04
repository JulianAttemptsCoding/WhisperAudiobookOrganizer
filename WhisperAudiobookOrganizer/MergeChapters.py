#!/usr/bin/env python3
import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_chapters.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def merge_chapter_files(input_directory: Path, output_directory: Path):
    """Merge chapter files into single chapter files"""
    
    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Input directory: {input_directory}")
    logger.info(f"Output directory: {output_directory}")
    
    # Find all mp3 files (both .mp3 and .Mp3)
    mp3_files_lower = list(input_directory.glob("*.mp3"))
    mp3_files_upper = list(input_directory.glob("*.Mp3"))
    mp3_files = mp3_files_lower + mp3_files_upper
    
    logger.info(f"Found {len(mp3_files)} MP3 files ({len(mp3_files_lower)} .mp3, {len(mp3_files_upper)} .Mp3)")
    
    if not mp3_files:
        logger.error("No MP3 files found in input directory")
        return
    
    # Group files by chapter number
    chapter_files = defaultdict(list)
    
    for file_path in mp3_files:
        filename = file_path.name
        logger.debug(f"Processing file: {filename}")
        
        # Extract chapter number from filename
        chapter_num = extract_chapter_number(filename)
        
        if chapter_num is not None:
            chapter_files[chapter_num].append(file_path)
            logger.debug(f"Assigned {filename} to chapter {chapter_num}")
        else:
            logger.warning(f"Could not determine chapter number for {filename}, skipping")
    
    # Sort files within each chapter
    for chapter_num in chapter_files:
        chapter_files[chapter_num].sort(key=lambda x: get_track_order(x.name))
    
    logger.info(f"Found {len(chapter_files)} chapters to merge")
    logger.info(f"Chapters found: {sorted(chapter_files.keys())}")
    
    # Log file assignments for debugging
    for chapter_num in sorted(chapter_files.keys()):
        files = [f.name for f in chapter_files[chapter_num]]
        logger.info(f"Chapter {chapter_num} files ({len(files)}): {files}")
    
    # Merge files for each chapter
    merged_count = 0
    for chapter_num in sorted(chapter_files.keys()):
        files = chapter_files[chapter_num]
        logger.info(f"Merging chapter {chapter_num}: {len(files)} files")
        
        if not files:
            continue
            
        # Create output filename
        output_filename = f"{chapter_num}.EOE.mp3"
        output_path = output_directory / output_filename
        
        # Merge files using ffmpeg
        if merge_files_with_ffmpeg(files, output_path):
            logger.info(f"Successfully merged chapter {chapter_num} -> {output_filename}")
            merged_count += 1
        else:
            logger.error(f"Failed to merge chapter {chapter_num}")
    
    logger.info(f"Successfully merged {merged_count} chapters")
    logger.info(f"Merged files are in: {output_directory}")

def extract_chapter_number(filename: str) -> int:
    """Extract chapter number from filename using multiple patterns"""
    try:
        # Remove file extension for processing
        name_without_ext = filename
        if filename.lower().endswith('.mp3'):
            name_without_ext = filename[:-4]
        
        logger.debug(f"Extracting chapter from: {filename} (without ext: {name_without_ext})")
        
        # Pattern 1: Files like "1.01", "2.02", etc.
        match = re.match(r'^(\d+)\.\d+', name_without_ext)
        if match:
            result = int(match.group(1))
            logger.debug(f"Pattern 1 matched: {result}")
            return result
        
        # Pattern 2: Files like "1.01 EOE", "2.02 EOE", etc.
        match = re.match(r'^(\d+)\.\d+\s+EOE', name_without_ext)
        if match:
            result = int(match.group(1))
            logger.debug(f"Pattern 2 matched: {result}")
            return result
        
        # Pattern 3: Files like "0 Introduction EOE", "1 Introduction EOE", etc.
        match = re.match(r'^(\d+)\s+.*', name_without_ext)
        if match:
            result = int(match.group(1))
            logger.debug(f"Pattern 3 matched: {result}")
            return result
            
        # Pattern 4: Files like "0.01 EOE", "1.01 EOE", etc.
        match = re.match(r'^(\d+)\.\d+\s+EOE', name_without_ext)
        if match:
            result = int(match.group(1))
            logger.debug(f"Pattern 4 matched: {result}")
            return result
            
        # Pattern 5: Files like "0.01", "1.01", etc.
        match = re.match(r'^(\d+)\.\d+', name_without_ext)
        if match:
            result = int(match.group(1))
            logger.debug(f"Pattern 5 matched: {result}")
            return result
            
        # Pattern 6: Any number at the beginning
        match = re.match(r'^(\d+)', name_without_ext)
        if match:
            result = int(match.group(1))
            logger.debug(f"Pattern 6 matched: {result}")
            return result
            
    except Exception as e:
        logger.warning(f"Error extracting chapter number from {filename}: {e}")
    
    logger.warning(f"Could not extract chapter number from {filename}")
    return None

def get_track_order(filename: str) -> int:
    """Get track order for sorting files within a chapter"""
    try:
        # Remove file extension for processing
        name_without_ext = filename
        if filename.lower().endswith('.mp3'):
            name_without_ext = filename[:-4]
        
        # Try to extract a number after the first dot
        if '.' in name_without_ext:
            parts = name_without_ext.split('.')
            if len(parts) > 1:
                second_part = parts[1]
                # Look for numbers at the beginning of the second part
                match = re.match(r'^(\d+)', second_part)
                if match:
                    return int(match.group(1))
        
        # Look for numbers after the first space
        if ' ' in name_without_ext:
            parts = name_without_ext.split(' ')
            if len(parts) > 1:
                second_part = parts[1]
                match = re.match(r'^(\d+)', second_part)
                if match:
                    return int(match.group(1))
                
        # Try to find any number in the filename
        numbers = re.findall(r'\d+', name_without_ext)
        if len(numbers) > 1:
            return int(numbers[1])  # Second number if available
        elif len(numbers) > 0:
            return int(numbers[0])   # First number
            
        # Default sorting
        return 0
        
    except Exception as e:
        logger.warning(f"Error getting track order for {filename}: {e}")
        return 0

def merge_files_with_ffmpeg(input_files: list, output_file: Path) -> bool:
    """Merge audio files using ffmpeg concatenation"""
    try:
        # Create a temporary file list for ffmpeg
        temp_dir = output_file.parent / "temp_merge"
        temp_dir.mkdir(exist_ok=True)
        file_list_path = temp_dir / "file_list.txt"
        
        # Write file list
        with open(file_list_path, 'w') as f:
            for file_path in input_files:
                # Use absolute paths and escape special characters
                absolute_path = file_path.absolute()
                # Escape single quotes in path
                escaped_path = str(absolute_path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        # Use ffmpeg to concatenate files
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(file_list_path),
            "-c", "copy",  # Copy codec (fast, no re-encoding)
            "-y",  # Overwrite output file
            str(output_file)
        ]
        
        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        # Clean up temp file
        try:
            file_list_path.unlink()
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")
        
        if result.returncode == 0 and output_file.exists():
            # Get total duration for logging
            file_size = output_file.stat().st_size
            logger.info(f"Created merged file: {output_file.name} ({file_size:,} bytes)")
            return True
        else:
            logger.error(f"FFmpeg failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"FFmpeg stderr: {result.stderr}")
            if result.stdout:
                logger.error(f"FFmpeg stdout: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timed out while merging files")
        return False
    except Exception as e:
        logger.error(f"Error merging files: {e}")
        logger.exception(e)
        return False

def main():
    """Main entry point"""
    if len(sys.argv) != 3:
        print("Usage: python3 merge_chapters.py <input_directory> <output_directory>")
        print("Example: python3 merge_chapters.py ~/output_transcriptions/renamed_mp3_files ~/merged_chapters")
        sys.exit(1)
    
    input_directory = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])
    
    if not input_directory.exists():
        logger.error(f"Input directory does not exist: {input_directory}")
        sys.exit(1)
    
    if not input_directory.is_dir():
        logger.error(f"Input path is not a directory: {input_directory}")
        sys.exit(1)
    
    try:
        merge_chapter_files(input_directory, output_directory)
    except KeyboardInterrupt:
        logger.info("Merging interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Merging failed: {e}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
