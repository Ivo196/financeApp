import argparse
import re
import sys
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    # Pattern to catch various YouTube URL formats
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def get_transcript(video_id, languages=['es', 'en']):
    """
    Fetches the transcript for a given video ID.
    Tries Spanish first, then English (or others provided).
    """
    try:
        # Instantiate the API
        yt_api = YouTubeTranscriptApi()
        # fetch returns the transcript data (list of dicts) directly
        transcript_list = yt_api.fetch(video_id, languages=languages)
        return transcript_list
    except VideoUnavailable:
        print(f"Error: The video {video_id} is unavailable.")
        return None
    except TranscriptsDisabled:
        print(f"Error: Transcripts are disabled for this video.")
        return None
    except NoTranscriptFound:
        print(f"Error: No transcript found for the requested languages ({languages}).")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def save_transcript(transcript, filename, interval_seconds=30):
    """
    Saves the transcript grouping text by time intervals to make it readable.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            current_group_start = 0
            current_text_block = []

            for entry in transcript:
                # Manejo de objeto vs dict
                start_time = entry.start if hasattr(entry, 'start') else entry['start']
                text = entry.text if hasattr(entry, 'text') else entry['text']
                
                # Limpiar saltos de línea dentro del texto de YouTube
                clean_text = text.replace('\n', ' ').strip()

                # Si el fragmento actual supera el intervalo, escribimos el bloque acumulado
                if start_time >= current_group_start + interval_seconds:
                    if current_text_block:
                        minutes = int(current_group_start) // 60
                        seconds = int(current_group_start) % 60
                        f.write(f"[{minutes:02d}:{seconds:02d}] {' '.join(current_text_block)}\n\n")
                    
                    # Reiniciamos para el siguiente bloque
                    current_group_start = (start_time // interval_seconds) * interval_seconds
                    current_text_block = [clean_text]
                else:
                    current_text_block.append(clean_text)

            # Escribir el último bloque restante
            if current_text_block:
                minutes = int(current_group_start) // 60
                seconds = int(current_group_start) % 60
                f.write(f"[{minutes:02d}:{seconds:02d}] {' '.join(current_text_block)}\n")

        print(f"Transcripción mejorada guardada en {filename}")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos using youtube-transcript-api.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output", help="Output filename", default="transcript.txt")
    parser.add_argument("-l", "--lang", help="Language codes (comma separated)", default="es,en")

    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    if not video_id:
        print("Error: Could not extract video ID from URL.")
        sys.exit(1)

    print(f"Fetching transcript for video ID: {video_id}...")
    languages = args.lang.split(',')
    
    transcript = get_transcript(video_id, languages=languages)
    
    if transcript:
        save_transcript(transcript, args.output)

if __name__ == "__main__":
    main()
