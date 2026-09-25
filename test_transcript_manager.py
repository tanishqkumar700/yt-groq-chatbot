from services.transcript_manager import TranscriptManager


VIDEO_ID = "dQw4w9WgXcQ"


def main():
    manager = TranscriptManager()

    try:
        transcript = manager.fetch_youtube_transcript(VIDEO_ID)

        print("Transcript fetched successfully.")
        print(f"Transcript length: {len(transcript)} characters")
        print("\nFirst 500 characters:\n")
        print(transcript[:500])

    except Exception as e:
        print(f"Transcript fetch failed: {e}")


if __name__ == "__main__":
    main()