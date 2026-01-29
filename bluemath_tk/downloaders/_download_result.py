from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DownloadResult:
    """
    Standardized result structure for download operations.

    This class provides a consistent interface for download results across all
    downloaders, making it easier to handle success/failure cases and track
    downloaded files.

    Attributes
    ----------
    success : bool
        Whether the download operation completed successfully.
    downloaded_files : List[str]
        List of file paths that were successfully downloaded.
    skipped_files : List[str]
        List of file paths that were skipped (e.g., already exist, incomplete).
    error_files : List[str]
        List of file paths that failed to download.
    errors : List[Dict[str, Any]]
        List of error dictionaries containing error details.
        Each dict has keys: 'file', 'error', 'timestamp'.
    metadata : Dict[str, Any]
        Additional metadata about the download operation.
    message : str
        Human-readable summary message.
    start_time : Optional[datetime]
        When the download operation started.
    end_time : Optional[datetime]
        When the download operation ended.
    duration_seconds : Optional[float]
        Total duration of the download operation in seconds.

    Examples
    --------
    >>> result = DownloadResult(
    ...     success=True,
    ...     downloaded_files=["/path/to/file1.nc", "/path/to/file2.nc"],
    ...     message="Downloaded 2 files successfully"
    ... )
    >>> print(result.message)
    Downloaded 2 files successfully
    >>> print(f"Success rate: {result.success_rate:.1%}")
    Success rate: 100.0%
    """

    success: bool = False
    downloaded_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    error_files: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def __post_init__(self):
        """Calculate duration if both start and end times are provided."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            self.duration_seconds = delta.total_seconds()

    @property
    def total_files(self) -> int:
        """Total number of files processed."""
        return (
            len(self.downloaded_files) + len(self.skipped_files) + len(self.error_files)
        )

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        if self.total_files == 0:
            return 0.0
        return len(self.downloaded_files) / self.total_files

    @property
    def has_errors(self) -> bool:
        """Whether any errors occurred."""
        return len(self.error_files) > 0 or len(self.errors) > 0

    def add_error(
        self, file_path: str, error: Exception, context: Dict[str, Any] = None
    ):
        """
        Add an error to the result.

        Parameters
        ----------
        file_path : str
            Path to the file that caused the error.
        error : Exception
            The exception that occurred.
        context : Dict[str, Any], optional
            Additional context about the error.
        """
        error_dict = {
            "file": file_path,
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.now().isoformat(),
        }
        if context:
            error_dict["context"] = context
        self.errors.append(error_dict)
        if file_path not in self.error_files:
            self.error_files.append(file_path)

    def add_downloaded(self, file_path: str):
        """Add a successfully downloaded file."""
        if file_path not in self.downloaded_files:
            self.downloaded_files.append(file_path)

    def add_skipped(self, file_path: str, reason: str = ""):
        """
        Add a skipped file.

        Parameters
        ----------
        file_path : str
            Path to the skipped file.
        reason : str, optional
            Reason why the file was skipped.
        """
        if file_path not in self.skipped_files:
            self.skipped_files.append(file_path)
        if reason:
            self.metadata.setdefault("skip_reasons", {})[file_path] = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "success": self.success,
            "downloaded_files": self.downloaded_files,
            "skipped_files": self.skipped_files,
            "error_files": self.error_files,
            "errors": self.errors,
            "metadata": self.metadata,
            "message": self.message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "total_files": self.total_files,
            "success_rate": self.success_rate,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.message:
            return self.message
        return (
            f"DownloadResult(success={self.success}, "
            f"downloaded={len(self.downloaded_files)}, "
            f"skipped={len(self.skipped_files)}, "
            f"errors={len(self.error_files)})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        duration = f"{self.duration_seconds:.1f}s" if self.duration_seconds else "N/A"
        return (
            f"DownloadResult(\n"
            f"  success={self.success},\n"
            f"  downloaded_files={len(self.downloaded_files)} files,\n"
            f"  skipped_files={len(self.skipped_files)} files,\n"
            f"  error_files={len(self.error_files)} files,\n"
            f"  total_files={self.total_files},\n"
            f"  success_rate={self.success_rate:.1%},\n"
            f"  duration={duration},\n"
            f")"
        )
