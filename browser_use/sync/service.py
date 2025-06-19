"""
Cloud sync service for sending events to the Browser Use cloud.
"""

import asyncio
import json
import logging
import os

import anyio
import httpx
from bubus import BaseEvent

from browser_use.sync.auth import TEMP_USER_ID, DeviceAuthClient

logger = logging.getLogger(__name__)


class CloudSync:
	"""Service for syncing events to the Browser Use cloud"""

	def __init__(self, base_url: str | None = None, enable_auth: bool = True):
		# Backend API URL for all API requests - can be passed directly or defaults to env var
		self.base_url = base_url or os.getenv('BROWSER_USE_CLOUD_URL', 'https://cloud.browser-use.com')
		self.enable_auth = enable_auth
		self.auth_client = DeviceAuthClient(base_url=self.base_url) if enable_auth else None
		self.pending_events: list[dict] = []
		self.auth_task = None
		self.session_id: str | None = None

	async def handle_event(self, event: BaseEvent) -> None:
		"""Handle an event by sending it to the cloud"""
		try:
			# Extract session ID from CreateAgentSessionEvent
			if event.event_type == 'CreateAgentSession' and hasattr(event, 'id'):
				self.session_id = event.id

				# Start authentication flow if enabled and not authenticated
				if self.enable_auth and self.auth_client and not self.auth_client.is_authenticated:
					# Start auth in background
					self.auth_task = asyncio.create_task(self._background_auth(agent_session_id=self.session_id))

			# Prepare event data
			event_data = self._prepare_event_data(event)

			# Send event to cloud
			await self._send_event(event_data)

		except Exception as e:
			logger.warning(f'Failed to send event to cloud: {e}')

	def _prepare_event_data(self, event: BaseEvent) -> dict:
		"""Prepare event data for cloud API"""
		# Get user_id from auth client or use temp ID
		user_id = self.auth_client.user_id if self.auth_client else TEMP_USER_ID

		# Update event with user_id
		event_dict = event.model_dump(mode='json')  # Use JSON mode to handle datetime serialization
		event_dict['user_id'] = user_id

		# Prepare API request format
		return {
			'event_type': event.event_type,
			'event_id': event_dict.get('event_id', event_dict.get('id', '')),
			'event_at': event_dict.get('event_at', event_dict.get('created_at', '')),
			'event_schema': f'{event.__class__.__name__}@1.0',
			'data': event_dict,
		}

	async def _send_event(self, event_data: dict) -> None:
		"""Send event to cloud API"""
		headers = {}

		# Add auth headers if available
		if self.auth_client:
			headers.update(self.auth_client.get_headers())

		# Send event
		async with httpx.AsyncClient() as client:
			response = await client.post(
				f'{self.base_url.rstrip("/")}/api/v1/events/',
				json={'events': [event_data]},
				headers=headers,
				timeout=10.0,
			)

			if response.status_code == 401 and self.auth_client and not self.auth_client.is_authenticated:
				# Store event for retry after auth
				self.pending_events.append(event_data)
			else:
				response.raise_for_status()

	async def _background_auth(self, agent_session_id: str) -> None:
		"""Run authentication in background"""
		try:
			# Run authentication
			success = await self.auth_client.authenticate(
				agent_session_id=agent_session_id,
				show_instructions=True,
			)

			if success:
				# Resend any pending events
				await self._resend_pending_events()

				# Update WAL events with real user_id
				await self._update_wal_user_ids(agent_session_id)

		except Exception as e:
			logger.warning(f'Background authentication failed: {e}')

	async def _resend_pending_events(self) -> None:
		"""Resend events that were queued during auth"""
		if not self.pending_events:
			return

		# Update user_id in pending events
		user_id = self.auth_client.user_id
		for event_data in self.pending_events:
			event_data['data']['user_id'] = user_id

		# Send all pending events
		for event_data in self.pending_events:
			try:
				await self._send_event(event_data)
			except Exception as e:
				logger.warning(f'Failed to resend pending event: {e}')

		self.pending_events.clear()

	async def _update_wal_user_ids(self, session_id: str) -> None:
		"""Update user IDs in WAL file after authentication"""
		try:
			from browser_use.utils import BROWSER_USE_CONFIG_DIR

			wal_path = BROWSER_USE_CONFIG_DIR / 'events' / f'{session_id}.jsonl'
			if not await anyio.Path(wal_path).exists():
				return

			# Read all events
			events = []
			content = await anyio.Path(wal_path).read_text()
			for line in content.splitlines():
				if line.strip():
					events.append(json.loads(line))

			# Update user_id
			user_id = self.auth_client.user_id
			for event in events:
				if 'user_id' in event:
					event['user_id'] = user_id

			# Write back
			updated_content = '\n'.join(json.dumps(event) for event in events) + '\n'
			await anyio.Path(wal_path).write_text(updated_content)

		except Exception as e:
			logger.warning(f'Failed to update WAL user IDs: {e}')

	async def wait_for_auth(self) -> None:
		"""Wait for authentication to complete if in progress"""
		if self.auth_task and not self.auth_task.done():
			await self.auth_task

	async def send_event(self, event_type: str, event_data: dict, event_schema: str) -> None:
		"""Send a single event to the cloud - convenience method for tests"""
		# Create a dynamic subclass that includes the custom event data
		from pydantic import ConfigDict, Field

		etype = event_type

		class CustomEvent(BaseEvent):
			model_config = ConfigDict(
				extra='allow', arbitrary_types_allowed=True, validate_assignment=True, validate_default=True
			)
			event_type: str = Field(default=etype, frozen=True)

		CustomEvent.__name__ = etype
		await self.handle_event(CustomEvent(**event_data))

	async def authenticate(self, show_instructions: bool = True) -> bool:
		"""Authenticate with the cloud service"""
		if not self.auth_client:
			return False

		return await self.auth_client.authenticate(agent_session_id=self.session_id, show_instructions=show_instructions)
