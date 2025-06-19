"""Tests for CloudSync client machinery - retry logic, event handling, backend communication."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from pytest_httpserver import HTTPServer

from browser_use.sync.auth import TEMP_USER_ID, DeviceAuthClient
from browser_use.sync.service import CloudSync


@pytest.fixture
def temp_config_dir():
	"""Create temporary config directory for tests."""
	with tempfile.TemporaryDirectory() as tmpdir:
		temp_dir = Path(tmpdir) / '.config' / 'browseruse'
		temp_dir.mkdir(parents=True, exist_ok=True)

		# Temporarily replace the config dir
		import browser_use.sync.auth
		import browser_use.utils

		original_auth = getattr(browser_use.sync.auth, 'BROWSER_USE_CONFIG_DIR', None)
		original_utils = getattr(browser_use.utils, 'BROWSER_USE_CONFIG_DIR', None)

		browser_use.sync.auth.BROWSER_USE_CONFIG_DIR = temp_dir
		browser_use.utils.BROWSER_USE_CONFIG_DIR = temp_dir

		yield temp_dir

		# Restore original
		if original_auth:
			browser_use.sync.auth.BROWSER_USE_CONFIG_DIR = original_auth
		if original_utils:
			browser_use.utils.BROWSER_USE_CONFIG_DIR = original_utils


@pytest.fixture
async def http_client(httpserver: HTTPServer):
	"""Create a real HTTP client pointed at the test server."""
	async with httpx.AsyncClient(base_url=httpserver.url_for('')) as client:
		yield client


class TestCloudSyncInit:
	"""Test CloudSync initialization and configuration."""

	async def test_init_with_auth_enabled(self, temp_config_dir):
		"""Test CloudSync initialization with auth enabled."""
		# Set test environment variable
		with patch.dict(os.environ, {'BROWSER_USE_CLOUD_URL': 'http://localhost:8000'}):
			service = CloudSync(enable_auth=True)

			assert service.base_url == 'http://localhost:8000'
			assert service.enable_auth is True
			assert service.auth_client is not None
			assert isinstance(service.auth_client, DeviceAuthClient)
			assert service.pending_events == []
			assert service.session_id is None

	async def test_init_with_auth_disabled(self, temp_config_dir):
		"""Test CloudSync initialization with auth disabled."""
		with patch.dict(os.environ, {'BROWSER_USE_CLOUD_URL': 'http://localhost:8000'}):
			service = CloudSync(enable_auth=False)

			assert service.base_url == 'http://localhost:8000'
			assert service.enable_auth is False
			assert service.auth_client is None
			assert service.pending_events == []


class TestCloudSyncEventHandling:
	"""Test CloudSync event validation and processing."""

	@pytest.fixture
	def authenticated_sync(self, httpserver: HTTPServer, http_client, temp_config_dir):
		"""Create authenticated CloudSync service."""
		auth = DeviceAuthClient(base_url=httpserver.url_for(''), http_client=http_client)
		auth.auth_config.api_token = 'test-api-key'
		auth.auth_config.user_id = 'test-user-123'

		service = CloudSync(base_url=httpserver.url_for(''), enable_auth=True)
		service.auth_client = auth
		service.session_id = 'test-session-id'
		return service

	@pytest.fixture
	def unauthenticated_sync(self, httpserver: HTTPServer, temp_config_dir):
		"""Create unauthenticated CloudSync service."""
		service = CloudSync(base_url=httpserver.url_for(''), enable_auth=True)
		service.session_id = 'test-session-id'
		return service

	async def test_event_forwarding_authenticated(self, httpserver: HTTPServer, authenticated_sync):
		"""Test event forwarding when authenticated."""
		# Capture requests
		requests = []

		def capture_request(request):
			requests.append(request.get_json())
			from werkzeug.wrappers import Response

			return Response('{"processed": 1, "failed": 0}', status=200, mimetype='application/json')

		httpserver.expect_request('/api/v1/events/', method='POST').respond_with_handler(capture_request)

		# Send event
		await authenticated_sync.send_event(
			event_type='CreateAgentTaskEvent',
			event_data={'task': 'Test task', 'priority': 'high'},
			event_schema='AgentTaskModel@1.0',
		)

		# Verify forwarding
		assert len(requests) == 1
		event_batch = requests[0]
		assert len(event_batch['events']) == 1

		event = event_batch['events'][0]
		assert event['event_type'] == 'CreateAgentTaskEvent'
		assert event['data']['user_id'] == 'test-user-123'
		# BaseEvent creates event_type attribute, plus our custom data as attributes
		assert event['data']['task'] == 'Test task'
		assert event['data']['priority'] == 'high'

	async def test_event_queueing_unauthenticated(self, httpserver: HTTPServer, unauthenticated_sync):
		"""Test event queueing when unauthenticated."""
		# Server returns 401
		httpserver.expect_request('/api/v1/events/', method='POST').respond_with_json({'error': 'unauthorized'}, status=401)

		# Send event
		await unauthenticated_sync.send_event(
			event_type='CreateAgentTaskEvent',
			event_data={'task': 'Queued task'},
			event_schema='AgentTaskModel@1.0',
		)

		# Event should be queued
		assert len(unauthenticated_sync.pending_events) == 1
		queued_event = unauthenticated_sync.pending_events[0]
		assert queued_event['event_type'] == 'CreateAgentTaskEvent'
		assert queued_event['data']['user_id'] == TEMP_USER_ID
		assert queued_event['data']['task'] == 'Queued task'

	async def test_event_user_id_injection_pre_auth(self, httpserver: HTTPServer, unauthenticated_sync):
		"""Test that temp user ID is injected for pre-auth events."""
		requests = []

		def capture_request(request):
			requests.append(request.get_json())
			from werkzeug.wrappers import Response

			return Response('{"processed": 1, "failed": 0}', status=200, mimetype='application/json')

		httpserver.expect_request('/api/v1/events/', method='POST').respond_with_handler(capture_request)

		# Send event without user_id
		await unauthenticated_sync.send_event(
			event_type='CreateAgentTaskEvent',
			event_data={'task': 'Pre-auth task'},
			event_schema='AgentTaskModel@1.0',
		)

		# Verify temp user ID was injected
		assert len(requests) == 1
		event = requests[0]['events'][0]
		assert event['data']['user_id'] == TEMP_USER_ID


class TestCloudSyncRetryLogic:
	"""Test CloudSync retry and error handling logic."""

	@pytest.fixture
	def sync_with_auth(self, httpserver: HTTPServer, http_client, temp_config_dir):
		"""Create CloudSync with auth."""
		auth = DeviceAuthClient(base_url=httpserver.url_for(''), http_client=http_client)
		auth.auth_config.api_token = 'test-api-key'
		auth.auth_config.user_id = 'test-user-123'

		service = CloudSync(base_url=httpserver.url_for(''), enable_auth=True)
		service.auth_client = auth
		service.session_id = 'test-session-id'
		return service

	async def test_pending_event_resending(self, httpserver: HTTPServer, sync_with_auth):
		"""Test resending of pending events after authentication."""
		requests = []

		def capture_request(request):
			requests.append(request.get_json())
			from werkzeug.wrappers import Response

			return Response('{"processed": 1, "failed": 0}', status=200, mimetype='application/json')

		httpserver.expect_request('/api/v1/events/', method='POST').respond_with_handler(capture_request)

		# Manually add pending events (simulating 401 scenario)
		sync_with_auth.pending_events.extend(
			[
				{
					'event_type': 'CreateAgentTaskEvent',
					'data': {'task': 'Pending task 1', 'user_id': TEMP_USER_ID},
				},
				{
					'event_type': 'CreateAgentTaskEvent',
					'data': {'task': 'Pending task 2', 'user_id': TEMP_USER_ID},
				},
			]
		)

		# Resend pending events
		await sync_with_auth._resend_pending_events()

		# Should send all pending events with updated user ID
		assert len(requests) == 2
		for i, request in enumerate(requests):
			event = request['events'][0]
			assert event['data']['user_id'] == 'test-user-123'  # Updated from temp ID
			assert f'Pending task {i + 1}' == event['data']['task']

		# Pending events should be cleared
		assert len(sync_with_auth.pending_events) == 0

	async def test_backend_error_resilience(self, httpserver: HTTPServer, sync_with_auth):
		"""Test resilience to backend errors."""
		# Server returns 500 error
		httpserver.expect_request('/api/v1/events/', method='POST').respond_with_data('Internal Server Error', status=500)

		# Should not raise exception
		await sync_with_auth.send_event(
			event_type='CreateAgentTaskEvent',
			event_data={'task': 'Task during outage'},
			event_schema='AgentTaskModel@1.0',
		)

		# Events should not be queued for 500 errors (only 401)
		assert len(sync_with_auth.pending_events) == 0

	async def test_network_error_resilience(self, sync_with_auth):
		"""Test resilience to network errors."""
		# No server running - will get connection error
		sync_with_auth.base_url = 'http://localhost:99999'  # Invalid port

		# Should not raise exception
		await sync_with_auth.send_event(
			event_type='CreateAgentTaskEvent',
			event_data={'task': 'Task during network error'},
			event_schema='AgentTaskModel@1.0',
		)

		# Should handle gracefully without crashing

	async def test_concurrent_event_sending(self, httpserver: HTTPServer, sync_with_auth):
		"""Test handling of concurrent event sending."""
		import asyncio

		requests = []

		def capture_request(request):
			requests.append(request.get_json())
			from werkzeug.wrappers import Response

			return Response('{"processed": 1, "failed": 0}', status=200, mimetype='application/json')

		httpserver.expect_request('/api/v1/events/', method='POST').respond_with_handler(capture_request)

		# Send multiple events concurrently
		tasks = []
		for i in range(5):
			task = sync_with_auth.send_event(
				event_type='CreateAgentTaskEvent',
				event_data={'task': f'Concurrent task {i}'},
				event_schema='AgentTaskModel@1.0',
			)
			tasks.append(task)

		await asyncio.gather(*tasks)

		# All events should be sent
		assert len(requests) == 5
		# Just verify all events have task data - order may vary due to concurrency
		task_values = [req['events'][0]['data']['task'] for req in requests]
		expected_tasks = [f'Concurrent task {i}' for i in range(5)]
		assert sorted(task_values) == sorted(expected_tasks)


class TestCloudSyncBackendCommunication:
	"""Test CloudSync backend communication patterns."""

	async def test_request_format_validation(self, httpserver: HTTPServer, temp_config_dir):
		"""Test that requests are formatted correctly for backend."""
		requests = []

		def capture_request(request):
			# Validate request structure
			assert request.content_type == 'application/json'
			data = request.get_json()
			requests.append(data)

			# Validate batch structure
			assert 'events' in data
			assert isinstance(data['events'], list)
			assert len(data['events']) == 1

			event = data['events'][0]
			required_fields = ['event_type', 'event_id', 'event_at', 'event_schema', 'data']
			for field in required_fields:
				assert field in event, f'Missing required field: {field}'

			from werkzeug.wrappers import Response

			return Response('{"processed": 1, "failed": 0}', status=200, mimetype='application/json')

		httpserver.expect_request('/api/v1/events/', method='POST').respond_with_handler(capture_request)

		# Create authenticated service
		auth = DeviceAuthClient(base_url=httpserver.url_for(''))
		auth.auth_config.api_token = 'test-api-key'
		auth.auth_config.user_id = 'test-user-123'

		service = CloudSync(base_url=httpserver.url_for(''), enable_auth=True)
		service.auth_client = auth
		service.session_id = 'test-session-id'

		await service.send_event(
			event_type='CreateAgentTaskEvent',
			event_data={'task': 'Format validation test'},
			event_schema='AgentTaskModel@1.0',
		)

		assert len(requests) == 1

	async def test_auth_header_handling(self, httpserver: HTTPServer, temp_config_dir):
		"""Test proper auth header handling."""
		requests = []

		def capture_request(request):
			requests.append(
				{
					'headers': dict(request.headers),
					'json': request.get_json(),
				}
			)
			from werkzeug.wrappers import Response

			return Response('{"processed": 1, "failed": 0}', status=200, mimetype='application/json')

		httpserver.expect_request('/api/v1/events/', method='POST').respond_with_handler(capture_request)

		# Test authenticated request
		auth = DeviceAuthClient(base_url=httpserver.url_for(''))
		auth.auth_config.api_token = 'test-api-key'
		auth.auth_config.user_id = 'test-user-123'

		service = CloudSync(base_url=httpserver.url_for(''), enable_auth=True)
		service.auth_client = auth

		await service.send_event(
			event_type='CreateAgentTaskEvent',
			event_data={'task': 'Auth header test'},
			event_schema='AgentTaskModel@1.0',
		)

		# Check auth header was included
		assert len(requests) == 1
		headers = requests[0]['headers']
		assert 'Authorization' in headers
		assert headers['Authorization'] == 'Bearer test-api-key'

		# Test unauthenticated request
		requests.clear()
		service.auth_client = DeviceAuthClient(base_url=httpserver.url_for(''))  # No credentials

		await service.send_event(
			event_type='CreateAgentTaskEvent',
			event_data={'task': 'No auth test'},
			event_schema='AgentTaskModel@1.0',
		)

		# Check no auth header
		assert len(requests) == 1
		headers = requests[0]['headers']
		assert 'Authorization' not in headers
