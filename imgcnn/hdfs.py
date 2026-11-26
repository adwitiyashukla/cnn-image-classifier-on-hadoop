import json
from urllib.parse import quote

import requests


class HDFS:
    def __init__(self, url, user):
        self.base = url.rstrip("/") + "/webhdfs/v1"
        self.user = user
        self.http = requests.Session()
        self.http.trust_env = False

    def _url(self, path):
        return self.base + quote(path)

    def _call(self, method, path, op, ok=(200,), **params):
        response = self.http.request(method, self._url(path), params={"op": op, "user.name": self.user, **params})
        if response.status_code not in ok:
            raise OSError(f"webhdfs {op} {path}: {response.status_code} {response.text[:300]}")
        return response

    def status(self, path):
        response = self._call("GET", path, "GETFILESTATUS", ok=(200, 404))
        return None if response.status_code == 404 else response.json()["FileStatus"]

    def list(self, path):
        return self._call("GET", path, "LISTSTATUS").json()["FileStatuses"]["FileStatus"]

    def files(self, path):
        found = []
        for entry in self.list(path):
            child = f"{path.rstrip('/')}/{entry['pathSuffix']}"
            if entry["type"] == "DIRECTORY":
                found += self.files(child)
            else:
                found.append((child, entry["length"]))
        return sorted(found)

    def mkdirs(self, path):
        self._call("PUT", path, "MKDIRS")

    def delete(self, path, recursive=True):
        self._call("DELETE", path, "DELETE", recursive=str(recursive).lower())

    def rename(self, source, target):
        if not self._call("PUT", source, "RENAME", destination=target).json()["boolean"]:
            raise OSError(f"webhdfs RENAME {source} to {target} was refused")

    def create(self, path, data, overwrite=True):
        response = self._call("PUT", path, "CREATE", overwrite=str(overwrite).lower(), noredirect="true")
        location = response.json()["Location"]
        upload = self.http.put(location, data=data, headers={"Content-Type": "application/octet-stream"})
        if upload.status_code != 201:
            raise OSError(f"webhdfs CREATE {path}: {upload.status_code} {upload.text[:300]}")

    def open(self, path, chunk=1 << 20):
        response = self.http.get(self._url(path), params={"op": "OPEN", "user.name": self.user}, stream=True)
        if response.status_code != 200:
            raise OSError(f"webhdfs OPEN {path}: {response.status_code} {response.text[:300]}")
        with response:
            yield from response.iter_content(chunk)

    def lines(self, path):
        pending = b""
        for chunk in self.open(path):
            pending += chunk
            *complete, pending = pending.split(b"\n")
            yield from complete
        if pending:
            yield pending

    def read_bytes(self, path):
        return b"".join(self.open(path))

    def read_json(self, path):
        return json.loads(self.read_bytes(path))

    def write_json(self, path, value):
        self.create(path, json.dumps(value, indent=1, sort_keys=True).encode())
