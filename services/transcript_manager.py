from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    InvalidVideoId,
)


class TranscriptError(Exception):
    """Base exception for transcript acquisition errors."""


class CaptionsUnavailableError(TranscriptError):
    """Raised when YouTube captions cannot be obtained."""


class VideoUnavailableError(TranscriptError):
    """Raised when the YouTube video is unavailable."""


class InvalidVideoIdError(TranscriptError):
    """Raised when the YouTube video ID is invalid."""


class TranscriptFetchError(TranscriptError):
    """Raised when transcript retrieval fails unexpectedly."""


class TranscriptManager:

    def __init__(self):
        self.api = YouTubeTranscriptApi()

    def fetch_youtube_transcript(self, video_id: str) -> str:
        try:
            transcript_obj = self.api.fetch(video_id)

            transcript = " ".join(
                block.text for block in transcript_obj
            )

            if not transcript.strip():
                raise CaptionsUnavailableError(
                    "YouTube captions are unavailable for this video."
                )

            return transcript

        except (TranscriptsDisabled, NoTranscriptFound) as exc:
            raise CaptionsUnavailableError(
                "YouTube captions are unavailable for this video."
            ) from exc

        except VideoUnavailable as exc:
            raise VideoUnavailableError(
                "The YouTube video is unavailable."
            ) from exc

        except InvalidVideoId as exc:
            raise InvalidVideoIdError(
                "The YouTube video ID is invalid."
            ) from exc

        except TranscriptError:
            raise

        except Exception as exc:
            raise TranscriptFetchError(
                "Unable to fetch the YouTube transcript."
            ) from exc