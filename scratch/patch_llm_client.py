with open("main/utils/llm_client.py", "r", encoding="utf-8") as f:
    content = f.read()

target1 = "adapter: str = \"generator\",\n    ) -> dict[str, Any]:"
repl1 = "adapter: str = \"generator\",\n        timeout: Optional[float] = None,\n    ) -> dict[str, Any]:"

target2 = "resp = self.http_client.post(url, json=payload, timeout=float(self.settings.REMOTE_TIMEOUT_SECONDS))"
repl2 = "effective_timeout = float(timeout if timeout is not None else self.settings.REMOTE_TIMEOUT_SECONDS)\n            resp = self.http_client.post(url, json=payload, timeout=effective_timeout)"

assert target1 in content, "target1 not found"
assert target2 in content, "target2 not found"

content = content.replace(target1, repl1, 1).replace(target2, repl2, 1)
with open("main/utils/llm_client.py", "w", encoding="utf-8", newline="") as f:
    f.write(content)
print("Updated llm_client.py successfully")
