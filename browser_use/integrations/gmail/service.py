"""
Gmail API Service for Browser Use

Handles Gmail API authentication, email reading, and 2FA code extraction.
This service provides a clean interface for agents to interact with Gmail.
"""

# @file purpose: Gmail API service for email reading and 2FA code extraction

import os
import re
import time
import base64
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from browser_use.config import CONFIG

logger = logging.getLogger(__name__)


class GmailService:
	"""
	Gmail API service for email reading and 2FA code extraction.

	Provides functionality to:
	- Authenticate with Gmail API using OAuth2
	- Read recent emails with filtering
	- Extract 2FA codes from email content
	- Poll for new 2FA codes
	"""

	# Gmail API scopes
	SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

	def __init__(
		self,
		credentials_file: Optional[str] = None,
		token_file: Optional[str] = None,
		config_dir: Optional[str] = None,
		access_token: Optional[str] = None,
	):
		"""
		Initialize Gmail Service

		Args:
		    credentials_file: Path to OAuth credentials JSON from Google Cloud Console
		    token_file: Path to store/load access tokens
		    config_dir: Directory to store config files (defaults to browser-use config directory)
		    access_token: Direct access token (skips file-based auth if provided)
		"""
		# Set up configuration directory using browser-use's config system
		if config_dir is None:
			self.config_dir = CONFIG.BROWSER_USE_CONFIG_DIR
		else:
			self.config_dir = Path(config_dir).expanduser().resolve()

		# Ensure config directory exists (only if not using direct token)
		if access_token is None:
			self.config_dir.mkdir(parents=True, exist_ok=True)

		# Set up credential paths
		self.credentials_file = credentials_file or self.config_dir / 'gmail_credentials.json'
		self.token_file = token_file or self.config_dir / 'gmail_token.json'

		# Direct access token support
		self.access_token = access_token

		self.service = None
		self.creds = None
		self._authenticated = False

	def is_authenticated(self) -> bool:
		"""Check if Gmail service is authenticated"""
		return self._authenticated and self.service is not None

	async def authenticate(self) -> bool:
		"""
		Handle OAuth authentication and token management

		Returns:
		    bool: True if authentication successful, False otherwise
		"""
		try:
			logger.info('🔐 Authenticating with Gmail API...')

			# Check if using direct access token
			if self.access_token:
				logger.info('🔑 Using provided access token')
				# Create credentials from access token
				self.creds = Credentials(token=self.access_token, scopes=self.SCOPES)
				# Test token validity by building service
				self.service = build('gmail', 'v1', credentials=self.creds)
				self._authenticated = True
				logger.info('✅ Gmail API ready with access token!')
				return True

			# Original file-based authentication flow
			# Try to load existing tokens
			if os.path.exists(self.token_file):
				self.creds = Credentials.from_authorized_user_file(str(self.token_file), self.SCOPES)
				logger.debug('📁 Loaded existing tokens')

			# If no valid credentials, run OAuth flow
			if not self.creds or not self.creds.valid:
				if self.creds and self.creds.expired and self.creds.refresh_token:
					logger.info('🔄 Refreshing expired tokens...')
					self.creds.refresh(Request())
				else:
					logger.info('🌐 Starting OAuth flow...')
					if not os.path.exists(self.credentials_file):
						logger.error(
							f'❌ Gmail credentials file not found: {self.credentials_file}\n'
							'Please download it from Google Cloud Console:\n'
							'1. Go to https://console.cloud.google.com/\n'
							'2. APIs & Services > Credentials\n'
							'3. Download OAuth 2.0 Client JSON\n'
							f"4. Save as 'gmail_credentials.json' in {self.config_dir}/"
						)
						return False

					flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_file), self.SCOPES)
					# Use specific redirect URI to match OAuth credentials
					self.creds = flow.run_local_server(port=8080, open_browser=True)

				# Save tokens for next time
				with open(self.token_file, 'w') as token:
					token.write(self.creds.to_json())
				logger.info(f'💾 Tokens saved to {self.token_file}')

			# Build Gmail service
			self.service = build('gmail', 'v1', credentials=self.creds)
			self._authenticated = True
			logger.info('✅ Gmail API ready!')
			return True

		except Exception as e:
			logger.error(f'❌ Gmail authentication failed: {e}')
			return False

	async def get_recent_emails(self, max_results: int = 10, query: str = '', time_filter: str = '1h') -> List[Dict[str, Any]]:
		"""
		Get recent emails with optional query filter

		Args:
		    max_results: Maximum number of emails to fetch
		    query: Gmail search query (e.g., 'from:noreply@example.com')
		    time_filter: Time filter (e.g., '5m', '1h', '1d')

		Returns:
		    List of email dictionaries with parsed content
		"""
		if not self.is_authenticated():
			logger.error('❌ Gmail service not authenticated. Call authenticate() first.')
			return []

		try:
			# Add time filter to query if provided
			if time_filter and 'newer_than:' not in query:
				query = f'newer_than:{time_filter} {query}'.strip()

			logger.info(f'📧 Fetching {max_results} recent emails...')
			if query:
				logger.debug(f'🔍 Query: {query}')

			# Get message list
			assert self.service is not None
			results = self.service.users().messages().list(userId='me', maxResults=max_results, q=query).execute()

			messages = results.get('messages', [])
			if not messages:
				logger.info('📭 No messages found')
				return []

			logger.info(f'📨 Found {len(messages)} messages, fetching details...')

			# Get full message details
			emails = []
			for i, message in enumerate(messages, 1):
				logger.debug(f'📖 Reading email {i}/{len(messages)}...')

				full_message = self.service.users().messages().get(userId='me', id=message['id'], format='full').execute()

				email_data = self._parse_email(full_message)
				emails.append(email_data)

			return emails

		except HttpError as error:
			logger.error(f'❌ Gmail API error: {error}')
			return []
		except Exception as e:
			logger.error(f'❌ Unexpected error fetching emails: {e}')
			return []

	def _parse_email(self, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Parse Gmail message into readable format"""
		headers = {h['name']: h['value'] for h in message['payload']['headers']}

		return {
			'id': message['id'],
			'thread_id': message['threadId'],
			'subject': headers.get('Subject', ''),
			'from': headers.get('From', ''),
			'to': headers.get('To', ''),
			'date': headers.get('Date', ''),
			'timestamp': int(message['internalDate']),
			'body': self._extract_body(message['payload']),
			'raw_message': message,
		}

	def _extract_body(self, payload: Dict[str, Any]) -> str:
		"""Extract email body from payload"""
		body = ''

		if payload.get('body', {}).get('data'):
			# Simple email body
			body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
		elif payload.get('parts'):
			# Multi-part email
			for part in payload['parts']:
				if part['mimeType'] == 'text/plain' and part.get('body', {}).get('data'):
					part_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
					body += part_body
				elif part['mimeType'] == 'text/html' and not body and part.get('body', {}).get('data'):
					# Fallback to HTML if no plain text
					body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')

		return body

	async def find_2fa_codes(
		self, time_filter: str = '5m', sender_email: Optional[str] = None, custom_patterns: Optional[List[str]] = None
	) -> List[Dict[str, Any]]:
		"""
		Find 2FA verification codes in recent emails

		Args:
		    time_filter: Time filter for emails (e.g., '5m', '1h', '1d')
		    sender_email: Filter by specific sender email
		    custom_patterns: Additional regex patterns to look for codes

		Returns:
		    List of emails containing potential 2FA codes
		"""
		logger.info(f'🔍 Looking for 2FA codes in emails from last {time_filter}...')

		# Build search query for 2FA-related emails - very inclusive with simple terms
		query_parts = [
			f'newer_than:{time_filter}',
			'(code OR otp OR "one time password" OR "verification code" OR 2FA OR "two-factor" OR "authentication code" OR verify OR "login code" OR "access your account" OR "sign in" OR "enter the code")',
		]

		if sender_email:
			query_parts.append(f'from:{sender_email}')

		query = ' '.join(query_parts)
		logger.info(f'🔍 Gmail search query: {query}')

		emails = await self.get_recent_emails(max_results=10, query=query)

		codes_found = []

		for email in emails:
			codes = self._extract_codes_from_text(email['body'], custom_patterns)
			if codes:
				codes_found.append(
					{
						'subject': email['subject'],
						'from': email['from'],
						'timestamp': email['timestamp'],
						'codes': codes,
						'body_snippet': email['body'][:200] + '...' if len(email['body']) > 200 else email['body'],
						'email_id': email['id'],
					}
				)

		# Sort by timestamp (most recent first) and keep only the most recent per sender
		codes_found.sort(key=lambda x: x['timestamp'], reverse=True)

		# Group by sender and keep only the most recent from each
		seen_senders = set()
		filtered_codes = []
		for email_with_codes in codes_found:
			sender = email_with_codes['from']
			if sender not in seen_senders:
				seen_senders.add(sender)
				filtered_codes.append(email_with_codes)

		logger.info(f'🎯 Found {len(filtered_codes)} recent emails with 2FA codes (most recent per sender)')
		return filtered_codes

	def _extract_codes_from_text(self, text: str, custom_patterns: Optional[List[str]] = None) -> List[str]:
		"""Extract potential verification codes from text"""
		if not text:
			return []

		# Common 2FA code patterns - prioritize context-specific patterns to avoid URL noise
		patterns = [
			# High priority: Context-specific patterns
			r'(?:code|verification|OTP)[:\s]+(\d{4,8})',  # "code: 123456"
			r'(?:your|the)\s+(?:code|verification)\s+(?:is|code)[:\s]+(\d{4,8})',  # "your code is 123456"
			r'(?:enter|use)\s+(?:code|verification)[:\s]+(\d{4,8})',  # "enter code: 123456"
			r'(?:below|code)\.\s*(\d{4,8})',  # "code below. 12345"
			r'(?:emailCode=)(\d{4,8})',  # URL parameter like emailCode=77599
			# Lower priority: Standalone numbers (but exclude those in long URLs)
			r'(?<![\w\-])\d{5}(?![\w\-])',  # 5-digit codes (like 80565) not in URLs
			r'(?<![\w\-])\d{6}(?![\w\-])',  # 6-digit codes (most common) not in URLs
		]

		# Add custom patterns if provided
		if custom_patterns:
			patterns.extend(custom_patterns)

		found_codes_with_priority = []

		for i, pattern in enumerate(patterns):
			matches = re.findall(pattern, text, re.IGNORECASE)
			for match in matches:
				code = match if isinstance(match, str) else str(match)
				# Assign priority based on pattern order (earlier patterns = higher priority)
				priority = len(patterns) - i
				found_codes_with_priority.append((code, priority))

		# Filter out codes from long URLs (except emailCode parameter)
		filtered_codes_with_priority = []
		for code, priority in found_codes_with_priority:
			# Skip codes that are part of long URL UUIDs or tokens
			code_index = text.find(code)
			if code_index != -1:
				# Check surrounding context for URL patterns
				context_start = max(0, code_index - 30)
				context_end = min(len(text), code_index + len(code) + 30)
				context = text[context_start:context_end]

				# Skip if it's part of a UUID in a URL (like af6e5ca8-f2ae-4932-97d9)
				if re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-' + re.escape(code) + r'-[a-f0-9]{4}', context):
					continue

				# Special case: emailCode parameter gets highest priority
				if 'emailCode=' in context:
					priority = 1000

				filtered_codes_with_priority.append((code, priority))

		# Sort by priority (highest first) and remove duplicates
		filtered_codes_with_priority.sort(key=lambda x: x[1], reverse=True)

		# Remove duplicates while preserving priority order
		seen = set()
		unique_codes = []
		for code, priority in filtered_codes_with_priority:
			if code not in seen and 4 <= len(code) <= 8 and re.match(r'^\d+$', code):
				seen.add(code)
				unique_codes.append(code)

		# If no numeric codes found, allow alphanumeric but with stricter filtering
		if not unique_codes:
			for code, priority in filtered_codes_with_priority:
				if (
					code not in seen
					and 4 <= len(code) <= 8
					and re.match(r'^[A-Z0-9]+$', code, re.IGNORECASE)
					and not re.match(r'^[A-F0-9]{6}$', code)  # Exclude likely hex color codes
					and code
					not in ['header', 'padding', 'button', 'margin', 'border', 'center', 'hidden', 'normal', 'mobile', 'device']
				):
					seen.add(code)
					unique_codes.append(code)

		return unique_codes

	async def poll_for_2fa_code(
		self,
		sender_email: Optional[str] = None,
		max_wait_time: int = 60,
		check_interval: int = 5,
		custom_patterns: Optional[List[str]] = None,
	) -> Optional[str]:
		"""
		Poll for new 2FA code from specific sender

		Args:
		    sender_email: Email address to monitor (e.g., 'noreply@example.com')
		    max_wait_time: Maximum time to wait in seconds
		    check_interval: How often to check in seconds
		    custom_patterns: Additional regex patterns to look for codes

		Returns:
		    First 2FA code found, or None if timeout
		"""
		logger.info(f'⏳ Polling for 2FA code for {max_wait_time} seconds...')
		if sender_email:
			logger.info(f'📧 Monitoring emails from: {sender_email}')

		start_time = time.time()

		while (time.time() - start_time) < max_wait_time:
			codes_found = await self.find_2fa_codes(time_filter='1m', sender_email=sender_email, custom_patterns=custom_patterns)

			if codes_found:
				first_code = codes_found[0]['codes'][0]
				logger.info(f'🎉 Found 2FA code: {first_code}')
				return first_code

			elapsed = int(time.time() - start_time)
			logger.debug(f'⏱️  No codes yet, waiting {check_interval}s... ({elapsed}s elapsed)')
			time.sleep(check_interval)

		logger.warning('⏰ Timeout reached, no 2FA codes found')
		return None
