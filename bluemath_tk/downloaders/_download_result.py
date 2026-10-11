import json
import os
from collections import Counter
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

    def merge(self, other: "DownloadResult") -> "DownloadResult":
        """
        Add the files, errors and metadata of another result to this one.

        Dictionary metadata (e.g. skip reasons) are merged, other values replaced.

        Parameters
        ----------
        other : DownloadResult
            The result to merge into this one.

        Returns
        -------
        DownloadResult
            This result, updated.
        """
        for file_path in other.downloaded_files:
            self.add_downloaded(file_path)
        for file_path in other.skipped_files:
            if file_path not in self.skipped_files:
                self.skipped_files.append(file_path)
        for file_path in other.error_files:
            if file_path not in self.error_files:
                self.error_files.append(file_path)
        self.errors.extend(other.errors)
        for key, value in other.metadata.items():
            if isinstance(value, dict):
                self.metadata.setdefault(key, {}).update(value)
            else:
                self.metadata[key] = value
        return self

    def save(self, file_path: str) -> None:
        """
        Save the result as JSON (e.g. to keep a log of automated runs).

        Parameters
        ----------
        file_path : str
            Path of the JSON file to write.
        """
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary(self, max_errors: int = 20) -> str:
        """
        Multi-line report: counts, skip reasons, size to download and errors.

        Parameters
        ----------
        max_errors : int, optional
            Maximum number of errors listed. Default is 20.

        Returns
        -------
        str
            The report.
        """
        lines = [
            str(self),
            f"  downloaded: {len(self.downloaded_files)}",
            f"  skipped:    {len(self.skipped_files)}",
            f"  errors:     {len(self.error_files)}",
        ]
        reasons = Counter(self.metadata.get("skip_reasons", {}).values())
        for reason, count in reasons.most_common():
            lines.append(f"    {count:6d} x {reason}")

        remote_sizes = self.metadata.get("remote_sizes", {})
        pending = [
            path
            for path, reason in self.metadata.get("skip_reasons", {}).items()
            if "dry run" in reason
        ]
        if pending and remote_sizes:
            total = sum(remote_sizes.get(path, 0) for path in pending)
            lines.append(
                f"  to download: {total / 1e9:,.1f} GB in {len(pending)} files"
            )

        for error in self.errors[:max_errors]:
            lines.append(f"  ERROR {error['file']}: {error['error']}")
        if len(self.errors) > max_errors:
            lines.append(f"  ... and {len(self.errors) - max_errors} more errors")
        if self.duration_seconds is not None:
            lines.append(f"  duration: {self.duration_seconds / 3600:.2f} h")

        return "\n".join(lines)

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
