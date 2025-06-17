import abc
import glob
import json
import os

# import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

# take in different formats: json, jsonl, csv, tsv, txt, pickle, bytes
# do operations like: save, load, write, read, append, update, delete, copy


def grab_files(
    dir_path: str | Path, *exts: str, filename: str | None = None
) -> list[str]:
    """Grab files with specified extensions in the given directory path.

    Args:
        dir_path (str | Path): path to search for files

    Returns:
        list: list of files with specified extensions
    """
    if filename is None and not exts:
        raise ValueError("Please specify a filename or extension")

    files = []
    if exts:
        for ext in exts:
            if filename:
                files.extend(
                    glob.glob(
                        os.path.join(dir_path, f"**/{filename}{ext}"), recursive=True
                    )
                )
            else:
                files.extend(
                    glob.glob(os.path.join(dir_path, f"**/*{ext}"), recursive=True)
                )
    else:
        files.extend(
            glob.glob(os.path.join(dir_path, f"**/{filename}.*"), recursive=True)
        )
    return files


class BaseFile(abc.ABC):
    # Extension registry shared by all subclasses
    _registry = {}

    exts = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for ext in cls.exts:
            BaseFile._registry[ext.lower()] = cls

    def __init__(self, path):
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path})"

    def check_path(self) -> None:
        if not self.path.parent.exists():
            os.makedirs(self.path.parent)

    @abc.abstractmethod
    def _save(self, data: Any, *args: Any, **kwargs: Any) -> None: ...

    def save(self, data: Any, *args: Any, **kwargs: Any) -> None:
        self.check_path()
        self._save(data, *args, **kwargs)

    @abc.abstractmethod
    def _append(self, data: Any, *args: Any, **kwargs: Any) -> None: ...

    def append(self, data: Any, *args: Any, **kwargs: Any) -> None:
        self.check_path()
        self._append(data, *args, **kwargs)

    @abc.abstractmethod
    def load(self) -> Any: ...


# class FormatMeta(abc.ABCMeta):
#     def __new__(cls, name, bases, attrs):
#         new_class = super().__new__(cls, name, bases, attrs)
#         # print(f"registering {new_class} for extensions {attrs.get('exts', [])}")
#         # Register subclass for each extension it handles
#         for ext in attrs.get("exts", []):
#             extension_map[ext] = new_class
#         return new_class

#     def __call__(cls, path, *args, **kwargs):
#         # Only intercept instantiation for File, not for subclasses
#         if cls is File:
#             extension = Path(path).suffix if isinstance(path, str) else path.suffix
#             subclass = extension_map.get(extension)
#             if subclass is None:
#                 raise ValueError(f"Unsupported file extension: {extension}")
#             # return super(FormatMeta, subclass).__call__(path, *args, **kwargs)
#             return type.__call__(subclass, path, *args, **kwargs)
#         return type.__call__(cls, path, *args, **kwargs)
#         # return super().__call__(path, *args, **kwargs)


# class File(abc.ABC, metaclass=FormatMeta):
#     exts: list[str] = []

#     # def __new__(cls: Type[T], path: str | Path, *args: Any, **kwargs: Any) -> T:
#     #     # Extract file extension
#     #     extension = Path(path).suffix if isinstance(path, str) else path.suffix
#     #     # Find the appropriate subclass for this extension
#     #     subclass = extension_map.get(extension, None)
#     #     # print(f"Mapping subclass: {subclass}")
#     #     if subclass:
#     #         return super().__new__(subclass)
#     #     raise ValueError(f"Unsupported file format: {extension}")

#     def __init__(self, path: str | Path) -> None:
#         self._path = path if isinstance(path, Path) else Path(path)

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}({self.path})!r"

#     @property
#     def path(self) -> Path:
#         return self._path

#     @property
#     def parent(self) -> Path:
#         return self.path.parent

#     @property
#     def ext(self) -> str:
#         return self.path.suffix

#     @abc.abstractmethod
#     def save(self, data: Any, *args: Any, **kwargs: Any) -> None: ...

#     @abc.abstractmethod
#     def load(self, *args: Any, **kwargs: Any) -> Any: ...

#     def is_subdir(self, parent: str | Path | None = None) -> bool:
#         if not parent:
#             parent = self.parent

#         if isinstance(parent, str):
#             parent = Path(parent)

#         return (
#             self.path.exists()
#             and self.path.is_dir()
#             and self.path.is_relative_to(parent)
#         )

#     def get_subpath(
#         self, start: str | int = 0, end: str | int | None = None, suffix=False
#     ) -> Path:
#         if isinstance(start, str):
#             start = self.path.parts.index(start)
#         if isinstance(end, str):
#             end = self.path.parts.index(end)

#         if end is None:
#             if suffix:
#                 subpath = Path(*self.path.parts[start:])
#             else:
#                 subpath = Path(*self.path.parts[start:-1]) / self.path.stem
#         else:
#             subpath = Path(*self.path.parts[start:end])

#         return subpath

#     def copy_to(self, dst) -> None:
#         if not isinstance(dst, File):
#             dst = self.__class__(dst)
#         dst.check_path()
#         shutil.copyfile(self.path, dst.path)

#     def move_to(self, dst) -> None:
#         if not isinstance(dst, File):
#             dst = self.__class__(dst)
#         dst.check_path()
#         shutil.move(self.path, dst.path)

#     def delete(self):
#         try:
#             os.remove(self.path)
#         except OSError:
#             pass

#     def check_path(self):
#         if not self.path.parent.exists():
#             os.makedirs(self.path.parent)


class TextFile(BaseFile):
    exts = [".txt", ".log", ".out", ".md"]

    def _save(self, data: str) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(data)

    def load(self) -> str:
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read()

    def _append(self, new_data: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(new_data)


class JsonFile(BaseFile):
    exts = [".json"]

    def _save(self, data: dict | list | tuple, *args, **kwargs) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f, *args, **kwargs)

    def load(self, *args, **kwargs) -> dict | list | tuple:
        with open(self.path, "r") as f:
            return json.load(f, *args, **kwargs)

    def _append(self, new_data: dict | list, *args, **kwargs) -> None:
        data = self.load()

        if isinstance(data, list):
            data.append(new_data)
        elif isinstance(data, dict):
            data.update(new_data)
        else:
            raise TypeError("Incompatibale data type")

        self._save(data, *args, **kwargs)


class JsonlFile(BaseFile):
    exts = [".jsonl"]

    def _save(self, data: list, *args, **kwargs) -> None:
        with open(self.path, "w") as f:
            for item in data:
                f.write(json.dumps(item, *args, **kwargs) + "\n")

    def load(self, *args, **kwargs) -> list:
        with open(self.path, "r") as f:
            return [json.loads(line, *args, **kwargs) for line in f]

    def _append(self, new_data: dict | list | tuple, *args, **kwargs) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(new_data, *args, **kwargs) + "\n")


class CsvFile(BaseFile):
    exts = [".csv", ".tsv"]

    def _save(self, data: pd.DataFrame, *args, **kwargs) -> None:
        data.to_csv(self.path, *args, **kwargs)

    def load(self, *args, **kwargs):
        return pd.read_csv(self.path, *args, **kwargs)

    def _append(self, new_data: pd.DataFrame) -> None:
        new_data.to_csv(self.path, mode="a", header=False, index=False)


class File:
    def __new__(cls, path: str | Path) -> BaseFile:
        ext = Path(path).suffix.lower()
        fmt_class = BaseFile._registry.get(ext)

        if fmt_class is None:
            raise ValueError(f"No handler registered for extension: {ext}")

        return fmt_class(path)

    def __init__(self, path: str | Path) -> None:
        self._path = path if isinstance(path, Path) else Path(path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def parent(self) -> Path:
        return self.path.parent

    def is_subdir(self, parent: str | Path | None = None) -> bool:
        if not parent:
            parent = self.parent

        if isinstance(parent, str):
            parent = Path(parent)

        return (
            self.path.exists()
            and self.path.is_dir()
            and self.path.is_relative_to(parent)
        )

    def get_subpath(
        self,
        start: str | int = 0,
        end: str | int | None = None,
        suffix: bool | None = False,
    ) -> Path:
        if isinstance(start, str):
            start = self.path.parts.index(start)
        if isinstance(end, str):
            end = self.path.parts.index(end)

        if end is None:
            if suffix:
                subpath = Path(*self.path.parts[start:])
            else:
                subpath = Path(*self.path.parts[start:-1]) / self.path.stem
        else:
            subpath = Path(*self.path.parts[start:end])

        return subpath

    def copy_to(self, dst) -> None:
        if not isinstance(dst, File):
            dst = File(dst)
        shutil.copyfile(self.path, dst.path)

    def move_to(self, dst) -> None:
        if not isinstance(dst, File):
            dst = File(dst)
        shutil.move(self.path, dst.path)

    def delete(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass


if __name__ == "__main__":
    # print(extension_map)
    tfile = File("test.txt")
    print(tfile.path)
    tfile._save("")
    tfile.load()
