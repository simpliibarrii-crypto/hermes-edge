"""
Agent Memory — persistent multi-level memory for Hermes Edge.
Inspired by Nous Research Hermes Agent's MEMORY.md pattern.
"""
import json, logging, os
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    role: str  # "user", "assistant", "system", "tool", "observation"
    content: str
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

@dataclass
class Conversation:
    id: str
    title: str = ""
    entries: list[MemoryEntry] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    summary: str = ""

class AgentMemory:
    """Persistent memory with short-term (conversation) and long-term (facts) storage."""

    def __init__(self, memory_dir: str = "~/.hermes/memory"):
        self.memory_dir = Path(memory_dir).expanduser()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._long_term_file = self.memory_dir / "long_term.json"
        self._facts: dict[str, Any] = self._load_long_term()
        self.current_conversation: Conversation | None = None

    def _load_long_term(self) -> dict:
        if self._long_term_file.exists():
            try:
                return json.loads(self._long_term_file.read_text())
            except Exception:
                pass
        return {"user_preferences": {}, "learned_facts": [], "interaction_count": 0}

    def _save_long_term(self):
        self._long_term_file.write_text(json.dumps(self._facts, indent=2))

    def start_conversation(self, conv_id: str = "", title: str = "") -> Conversation:
        conv = Conversation(
            id=conv_id or datetime.now().strftime("conv_%Y%m%d_%H%M%S"),
            title=title,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self.current_conversation = conv
        self._save_conversation(conv)
        return conv

    def add_entry(self, role: str, content: str, metadata: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        if self.current_conversation:
            self.current_conversation.entries.append(entry)
            self.current_conversation.updated_at = datetime.now().isoformat()
            self._save_conversation(self.current_conversation)
        # Update interaction count
        self._facts["interaction_count"] += 1
        self._save_long_term()
        return entry

    def _save_conversation(self, conv: Conversation):
        path = self.memory_dir / f"{conv.id}.json"
        data = {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "summary": conv.summary,
            "entries": [{"role": e.role, "content": e.content, "timestamp": e.timestamp} for e in conv.entries],
        }
        path.write_text(json.dumps(data, indent=2))

    def load_conversation(self, conv_id: str) -> Conversation | None:
        path = self.memory_dir / f"{conv_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            conv = Conversation(
                id=data["id"], title=data.get("title", ""),
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
                summary=data.get("summary", ""),
            )
            for e in data.get("entries", []):
                conv.entries.append(MemoryEntry(
                    role=e["role"], content=e["content"],
                    timestamp=e.get("timestamp", ""),
                ))
            self.current_conversation = conv
            return conv
        except Exception as e:
            log.warning("Failed to load conversation %s: %s", conv_id, e)
            return None

    def list_conversations(self) -> list[dict]:
        results = []
        for f in sorted(self.memory_dir.glob("conv_*.json"), reverse=True)[:50]:
            try:
                data = json.loads(f.read_text())
                results.append({
                    "id": data["id"], "title": data.get("title", ""),
                    "updated_at": data.get("updated_at", ""),
                    "entry_count": len(data.get("entries", [])),
                })
            except Exception:
                pass
        return results

    def remember(self, key: str, value: Any):
        """Store a long-term fact."""
        self._facts["learned_facts"].append({
            "key": key, "value": value,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_long_term()

    def recall(self, key: str) -> list[Any]:
        """Retrieve long-term facts by key."""
        return [f["value"] for f in self._facts["learned_facts"] if f["key"] == key]

    def get_summary(self) -> str:
        """Return a summary of the current memory state for the system prompt."""
        facts = self._facts
        parts = [f"Interaction count: {facts['interaction_count']}"]
        if facts["user_preferences"]:
            parts.append(f"User preferences: {json.dumps(facts['user_preferences'])}")
        if facts["learned_facts"]:
            recent = facts["learned_facts"][-5:]
            parts.append("Learned facts: " + "; ".join(
                f"{f['key']}={f['value']}" for f in recent
            ))
        if self.current_conversation and self.current_conversation.entries:
            parts.append(f"Current conversation: {len(self.current_conversation.entries)} messages")
        return " | ".join(parts)

    def set_preference(self, key: str, value: Any):
        self._facts["user_preferences"][key] = value
        self._save_long_term()
