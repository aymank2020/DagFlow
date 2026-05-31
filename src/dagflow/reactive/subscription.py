"""Subscription management with auto-dispose and weak references.

Provides lifecycle management for stream subscriptions, including
automatic cleanup when subscribers are garbage collected, grouped
subscription management, and reference counting.

This module depends on:
- reactive.stream (ReactiveStream, StreamEvent) for subscription targets
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum, auto

from dagflow.reactive.stream import ReactiveStream, StreamEvent


class SubscriptionState(Enum):
    """Lifecycle state of a subscription."""

    ACTIVE = auto()
    PAUSED = auto()
    DISPOSED = auto()


@dataclass
class Subscription:
    """A managed subscription to a ReactiveStream.

    Tracks the lifecycle of a single listener registration, supporting
    pause/resume, disposal, and optional filtering of events.

    Attributes:
        sub_id: Unique subscription identifier.
        stream: The stream this subscription is attached to.
        listener_index: Index in the stream's listener list.
        state: Current lifecycle state.
        event_count: Number of events delivered through this subscription.
        filter_fn: Optional predicate to filter events before delivery.
    """

    sub_id: str
    stream: ReactiveStream
    listener_index: int
    state: SubscriptionState = SubscriptionState.ACTIVE
    event_count: int = 0
    filter_fn: Optional[Callable[[StreamEvent], bool]] = None
    _original_listener: Optional[Callable[[StreamEvent], None]] = field(
        default=None, repr=False
    )

    def pause(self) -> None:
        """Pause this subscription. Events will be silently dropped."""
        if self.state == SubscriptionState.ACTIVE:
            self.state = SubscriptionState.PAUSED

    def resume(self) -> None:
        """Resume a paused subscription."""
        if self.state == SubscriptionState.PAUSED:
            self.state = SubscriptionState.ACTIVE

    def dispose(self) -> None:
        """Permanently dispose this subscription. Cannot be resumed."""
        if self.state != SubscriptionState.DISPOSED:
            self.state = SubscriptionState.DISPOSED
            self.stream.unsubscribe(self.listener_index)

    @property
    def is_active(self) -> bool:
        """Whether this subscription is currently delivering events."""
        return self.state == SubscriptionState.ACTIVE

    @property
    def is_disposed(self) -> bool:
        """Whether this subscription has been permanently disposed."""
        return self.state == SubscriptionState.DISPOSED


class SubscriptionManager:
    """Manages multiple subscriptions with grouped lifecycle control.

    Provides a central registry for subscriptions, supporting bulk
    operations (pause all, dispose all), weak-reference tracking for
    automatic cleanup, and subscription groups.

    Usage:
        manager = SubscriptionManager()
        sub = manager.subscribe(stream, my_callback, group="ui")
        manager.pause_group("ui")
        manager.dispose_all()
    """

    def __init__(self) -> None:
        self._subscriptions: Dict[str, Subscription] = {}
        self._groups: Dict[str, Set[str]] = {}
        self._counter: int = 0
        self._weak_refs: Dict[str, weakref.ref] = {}

    @property
    def active_count(self) -> int:
        """Number of currently active subscriptions."""
        return sum(
            1 for s in self._subscriptions.values() if s.is_active
        )

    @property
    def total_count(self) -> int:
        """Total number of managed subscriptions (including disposed)."""
        return len(self._subscriptions)

    def subscribe(
        self,
        stream: ReactiveStream,
        listener: Callable[[StreamEvent], None],
        group: str = "default",
        filter_fn: Optional[Callable[[StreamEvent], bool]] = None,
        owner: Optional[Any] = None,
    ) -> Subscription:
        """Create a managed subscription to a stream.

        Args:
            stream: The ReactiveStream to subscribe to.
            listener: Callback for events.
            group: Group name for bulk operations.
            filter_fn: Optional predicate to filter events.
            owner: Optional owner object; subscription auto-disposes when
                   the owner is garbage collected.

        Returns:
            A Subscription handle for lifecycle control.
        """
        self._counter += 1
        sub_id = f"sub_{self._counter}"

        # Wrap listener with subscription-aware delivery
        sub_ref_holder: List[Optional[Subscription]] = [None]

        def managed_listener(event: StreamEvent) -> None:
            sub = sub_ref_holder[0]
            if sub is None or sub.state != SubscriptionState.ACTIVE:
                return
            if sub.filter_fn and not sub.filter_fn(event):
                return
            listener(event)
            sub.event_count += 1

        listener_index = stream.subscribe(managed_listener)

        sub = Subscription(
            sub_id=sub_id,
            stream=stream,
            listener_index=listener_index,
            filter_fn=filter_fn,
            _original_listener=listener,
        )
        sub_ref_holder[0] = sub

        self._subscriptions[sub_id] = sub

        # Group tracking
        if group not in self._groups:
            self._groups[group] = set()
        self._groups[group].add(sub_id)

        # Weak reference for auto-dispose
        if owner is not None:
            self._setup_weak_ref(sub_id, owner)

        return sub

    def unsubscribe(self, sub_id: str) -> bool:
        """Dispose a subscription by ID.

        Returns:
            True if the subscription was found and disposed.
        """
        sub = self._subscriptions.get(sub_id)
        if sub is None:
            return False
        sub.dispose()
        return True

    def pause_group(self, group: str) -> int:
        """Pause all subscriptions in a group.

        Returns:
            Number of subscriptions paused.
        """
        count = 0
        for sub_id in self._groups.get(group, set()):
            sub = self._subscriptions.get(sub_id)
            if sub and sub.is_active:
                sub.pause()
                count += 1
        return count

    def resume_group(self, group: str) -> int:
        """Resume all subscriptions in a group.

        Returns:
            Number of subscriptions resumed.
        """
        count = 0
        for sub_id in self._groups.get(group, set()):
            sub = self._subscriptions.get(sub_id)
            if sub and sub.state == SubscriptionState.PAUSED:
                sub.resume()
                count += 1
        return count

    def dispose_group(self, group: str) -> int:
        """Dispose all subscriptions in a group.

        Returns:
            Number of subscriptions disposed.
        """
        count = 0
        for sub_id in list(self._groups.get(group, set())):
            sub = self._subscriptions.get(sub_id)
            if sub and not sub.is_disposed:
                sub.dispose()
                count += 1
        return count

    def dispose_all(self) -> int:
        """Dispose all managed subscriptions.

        Returns:
            Number of subscriptions disposed.
        """
        count = 0
        for sub in self._subscriptions.values():
            if not sub.is_disposed:
                sub.dispose()
                count += 1
        return count

    def get_group_stats(self, group: str) -> Dict[str, int]:
        """Get statistics for a subscription group.

        Returns:
            Dict with counts of active, paused, and disposed subscriptions.
        """
        stats = {"active": 0, "paused": 0, "disposed": 0}
        for sub_id in self._groups.get(group, set()):
            sub = self._subscriptions.get(sub_id)
            if sub:
                if sub.state == SubscriptionState.ACTIVE:
                    stats["active"] += 1
                elif sub.state == SubscriptionState.PAUSED:
                    stats["paused"] += 1
                elif sub.state == SubscriptionState.DISPOSED:
                    stats["disposed"] += 1
        return stats

    def cleanup_disposed(self) -> int:
        """Remove disposed subscriptions from internal tracking.

        Returns:
            Number of entries cleaned up.
        """
        disposed_ids = [
            sid for sid, sub in self._subscriptions.items() if sub.is_disposed
        ]
        for sid in disposed_ids:
            del self._subscriptions[sid]
            for group_set in self._groups.values():
                group_set.discard(sid)
            self._weak_refs.pop(sid, None)
        return len(disposed_ids)

    def _setup_weak_ref(self, sub_id: str, owner: Any) -> None:
        """Set up a weak reference to auto-dispose when owner is collected."""

        def _on_owner_collected(ref: weakref.ref) -> None:
            sub = self._subscriptions.get(sub_id)
            if sub and not sub.is_disposed:
                sub.dispose()

        try:
            ref = weakref.ref(owner, _on_owner_collected)
            self._weak_refs[sub_id] = ref
        except TypeError:
            # Owner doesn't support weak references; skip auto-dispose
            pass
