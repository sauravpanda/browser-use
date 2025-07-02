"""
Gmail Integration for Browser Use

Provides Gmail API integration for handling 2FA codes and email reading.
This integration enables agents to authenticate via email when 2FA is required.

Usage:
    from browser_use.integrations.gmail import GmailService, register_gmail_actions

    # Option 1: Register Gmail actions with file-based authentication
    controller = Controller()
    register_gmail_actions(controller)

    # Option 2: Register Gmail actions with direct access token (recommended for production)
    controller = Controller()
    register_gmail_actions(controller, access_token="your_access_token_here")

    # Option 3: Use the service directly
    gmail = GmailService(access_token="your_access_token_here")
    await gmail.authenticate()
    codes = await gmail.find_2fa_codes()
"""

# @file purpose: Gmail integration for 2FA email authentication and email reading

from .service import GmailService
from .actions import register_gmail_actions

__all__ = ['GmailService', 'register_gmail_actions']
