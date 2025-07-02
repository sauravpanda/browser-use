"""
Gmail Actions for Browser Use

Defines agent actions for Gmail integration including 2FA code retrieval,
email reading, and authentication management.
"""

# @file purpose: Gmail actions for agent integration and 2FA code handling

import asyncio
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from browser_use.agent.views import ActionResult
from browser_use.controller.service import Controller
from .service import GmailService

logger = logging.getLogger(__name__)

# Global Gmail service instance - initialized when actions are registered
_gmail_service: Optional[GmailService] = None


class Find2FACodesParams(BaseModel):
	"""Parameters for finding 2FA codes in emails"""

	time_filter: str = Field(default='2m', description='Time filter for emails (e.g., "2m", "5m", "1h", "1d")')
	sender_email: Optional[str] = Field(default=None, description='Filter by specific sender email address')
	custom_patterns: Optional[List[str]] = Field(default=None, description='Additional regex patterns to search for codes')


class GetRecentEmailsParams(BaseModel):
	"""Parameters for getting recent emails"""

	max_results: int = Field(default=10, description='Maximum number of emails to fetch')
	query: str = Field(default='', description='Gmail search query (e.g., "from:noreply@example.com")')
	time_filter: str = Field(default='1h', description='Time filter for emails (e.g., "5m", "1h", "1d")')


def register_gmail_actions(
	controller: Controller, gmail_service: Optional[GmailService] = None, access_token: Optional[str] = None
) -> None:
	"""
	Register Gmail actions with the provided controller

	Args:
	    controller: The browser-use controller to register actions with
	    gmail_service: Optional pre-configured Gmail service instance
	    access_token: Optional direct access token (alternative to file-based auth)
	"""
	global _gmail_service

	# Use provided service or create a new one with access token if provided
	if gmail_service:
		_gmail_service = gmail_service
	elif access_token:
		_gmail_service = GmailService(access_token=access_token)
	else:
		_gmail_service = GmailService()

	@controller.action(
		description='🔐 **USE THIS FOR VERIFICATION CODES - NOT get_recent_emails** 🔐 '
		'When you need verification codes/OTP/2FA: USE THIS ACTION IMMEDIATELY! '
		'This searches Gmail, finds codes, extracts them automatically. '
		'DO NOT use get_recent_emails for verification - it cannot extract codes! '
		'This action is specifically built for code extraction from emails. '
		'One call gets you the actual numeric code ready to input.',
		param_model=Find2FACodesParams,
	)
	async def find_2fa_codes(params: Find2FACodesParams) -> ActionResult:
		"""Find 2FA verification codes in recent emails"""
		try:
			assert _gmail_service is not None, 'Gmail service not initialized'

			# Ensure authentication
			if not _gmail_service.is_authenticated():
				logger.info('📧 Gmail not authenticated, attempting authentication...')
				authenticated = await _gmail_service.authenticate()
				if not authenticated:
					return ActionResult(
						extracted_content='❌ Failed to authenticate with Gmail. Please ensure Gmail credentials are set up properly.',
						include_in_memory=True,
					)

			# Find 2FA codes with retry logic (4 attempts with longer waits for email delivery)
			max_retries = 4
			codes_found = None
			wait_times = [10, 20, 30]  # 10s, 20s, 30s (total 60s wait time)

			for attempt in range(max_retries):
				# Ensure authentication before each attempt
				if not _gmail_service.is_authenticated():
					logger.info(f'📧 Gmail not authenticated (attempt {attempt + 1}), attempting authentication...')
					authenticated = await _gmail_service.authenticate()
					if not authenticated:
						logger.warning(f'❌ Gmail authentication failed on attempt {attempt + 1}')
						# Continue to next attempt rather than failing immediately
						if attempt < max_retries - 1:
							wait_time = wait_times[attempt]
							logger.info(f'🔄 Retrying authentication in {wait_time}s...')
							await asyncio.sleep(wait_time)
						continue

				codes_found = await _gmail_service.find_2fa_codes(
					time_filter=params.time_filter, sender_email=params.sender_email, custom_patterns=params.custom_patterns
				)

				if codes_found:
					break

				# If not the last attempt, wait for email delivery
				if attempt < max_retries - 1:
					wait_time = wait_times[attempt]
					logger.info(f'🔄 No 2FA codes found (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...')
					await asyncio.sleep(wait_time)

			if not codes_found:
				return ActionResult(
					extracted_content=f'🔍 No 2FA codes found in emails from the last {params.time_filter} after {max_retries} attempts. '
					f'{"Filter: " + params.sender_email if params.sender_email else ""}',
					include_in_memory=True,
				)

			# Format results
			results = []
			for email_with_codes in codes_found:
				results.append(
					{
						'codes': email_with_codes['codes'],
						'from': email_with_codes['from'],
						'subject': email_with_codes['subject'],
						'snippet': email_with_codes['body_snippet'],
					}
				)

			# Return the most recent code as the primary result
			primary_code = codes_found[0]['codes'][0]

			content = f'✅ Found 2FA code: {primary_code}\n'
			content += f'📧 From: {codes_found[0]["from"]}\n'
			content += f'📝 Subject: {codes_found[0]["subject"]}\n'

			if len(codes_found) > 1:
				content += f'\n🔍 Additional {len(codes_found) - 1} emails with codes found.'

			return ActionResult(extracted_content=content, include_in_memory=True)

		except Exception as e:
			logger.error(f'Error finding 2FA codes: {e}')
			return ActionResult(extracted_content=f'❌ Error finding 2FA codes: {str(e)}', include_in_memory=True)

	@controller.action(
		description='📧 **General Email Reading** - For newsletters, notifications, and email content analysis. '
		'⚠️ DO NOT use when you need verification codes, OTP, 2FA, or login codes! '
		"For any verification scenario, use find_2fa_codes instead - it's specifically designed for that. "
		'This action has limited search capabilities that miss verification emails.',
		param_model=GetRecentEmailsParams,
	)
	async def get_recent_emails(params: GetRecentEmailsParams) -> ActionResult:
		"""Get recent emails from Gmail"""
		try:
			assert _gmail_service is not None, 'Gmail service not initialized'

			# Ensure authentication
			if not _gmail_service.is_authenticated():
				logger.info('📧 Gmail not authenticated, attempting authentication...')
				authenticated = await _gmail_service.authenticate()
				if not authenticated:
					return ActionResult(
						extracted_content='❌ Failed to authenticate with Gmail. Please ensure Gmail credentials are set up properly.',
						include_in_memory=True,
					)

			# Get emails
			emails = await _gmail_service.get_recent_emails(
				max_results=params.max_results, query=params.query, time_filter=params.time_filter
			)

			if not emails:
				query_info = f" matching '{params.query}'" if params.query else ''
				return ActionResult(
					extracted_content=f'📭 No emails found from the last {params.time_filter}{query_info}', include_in_memory=True
				)

			# Format email summary
			content = f'📧 Found {len(emails)} recent emails:\n\n'

			for i, email in enumerate(emails[:5], 1):  # Show first 5 emails
				content += f'{i}. From: {email["from"]}\n'
				content += f'   Subject: {email["subject"]}\n'
				content += f'   Preview: {email["body"][:100]}...\n\n'

			if len(emails) > 5:
				content += f'... and {len(emails) - 5} more emails'

			return ActionResult(extracted_content=content, include_in_memory=True)

		except Exception as e:
			logger.error(f'Error getting recent emails: {e}')
			return ActionResult(extracted_content=f'❌ Error getting recent emails: {str(e)}', include_in_memory=True)

	@controller.action(
		description='🔧 **Gmail Setup** - Establishes Gmail connection for email access. '
		'Usually auto-handled, but call this if email actions report authentication errors. '
		'Enables automatic retrieval of verification codes and email content.',
	)
	async def authenticate_gmail() -> ActionResult:
		"""Authenticate with Gmail API"""
		try:
			assert _gmail_service is not None, 'Gmail service not initialized'

			authenticated = await _gmail_service.authenticate()

			if authenticated:
				return ActionResult(
					extracted_content='✅ Successfully authenticated with Gmail API. Ready to read emails.',
					include_in_memory=True,
				)
			else:
				return ActionResult(
					extracted_content='❌ Failed to authenticate with Gmail. Please check your credentials setup:\n'
					'1. Download OAuth credentials from Google Cloud Console\n'
					"2. Save as 'gmail_credentials.json' in ~/.config/browseruse/\n"
					'3. Try again',
					include_in_memory=True,
				)

		except Exception as e:
			logger.error(f'Error authenticating Gmail: {e}')
			return ActionResult(extracted_content=f'❌ Error authenticating with Gmail: {str(e)}', include_in_memory=True)
