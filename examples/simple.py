import asyncio
import json
import os
import sys
from pathlib import Path

from browser_use.llm.openai.chat import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

# Set environment variable to enable debug logging to see intermediate prompts
os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'debug'

from browser_use import Agent
from browser_use.controller.service import Controller
from browser_use.integrations.gmail import GmailService, register_gmail_actions
from browser_use.logging_config import setup_logging

# Setup logging to see intermediate prompts
setup_logging()

# Initialize the model
llm = ChatOpenAI(
	model='gpt-4.1',
)

task = 'Go to allrecipes and login with username: hannahstone@halluminate.ai and password: 8ytRBo2BZX105nbpgUJF. Once logged in, get basic information and then logout.'

# Create a directory to save conversation history
conversation_dir = Path('./conversation_logs')
conversation_dir.mkdir(exist_ok=True)

# Create controller with Gmail actions
controller = Controller()

# Option 1: Use file-based Gmail authentication (default)
gmail_service = GmailService()

# Option 2: Use direct access token (recommended for production/CI environments)
gmail_access_token = os.getenv('GMAIL_ACCESS_TOKEN')
if gmail_access_token:
	gmail_service = GmailService(access_token=gmail_access_token)
	print('🔑 Using Gmail access token from environment variable')


# Pre-authenticate Gmail service
async def setup_gmail():
	print('🔐 Setting up Gmail authentication...')

	# Check if using access token
	if hasattr(gmail_service, 'access_token') and gmail_service.access_token:
		print('🔑 Using direct access token')
	else:
		print(f'📁 Looking for credentials at: {gmail_service.credentials_file}')

		if not os.path.exists(gmail_service.credentials_file):
			print(f"""
❌ Gmail credentials not found!
   
   To enable automatic 2FA code retrieval, choose one option:
   
   📁 Option A - File-based authentication (for development):
   1. Go to https://console.cloud.google.com/
   2. Enable Gmail API and create OAuth 2.0 credentials
   3. Download credentials JSON and save as: {gmail_service.credentials_file}
   
   🔑 Option B - Access token (for production/CI):
   1. Set environment variable: export GMAIL_ACCESS_TOKEN="your_token_here"
   2. Or pass directly: GmailService(access_token="your_token")
   
   Continuing without Gmail - you'll need to manually enter 2FA codes.
            """)
			return False

	try:
		authenticated = await gmail_service.authenticate()
		if authenticated:
			print('✅ Gmail authentication successful - 2FA codes will be retrieved automatically')
			# Test Gmail access
			print('🧪 Testing Gmail access...')
			test_emails = await gmail_service.get_recent_emails(max_results=1, query='', time_filter='1d')
			print(f'📧 Can access Gmail - found {len(test_emails)} recent emails')
			return True
		else:
			print("❌ Gmail authentication failed - 2FA codes won't be available")
			return False
	except Exception as e:
		print(f'❌ Gmail setup error: {e}')
		return False


register_gmail_actions(controller, gmail_service)

# Debug: Show available actions (commented out due to type issues)
# def show_available_actions():
#     print("\n🔧 AVAILABLE ACTIONS:")
#     # Will debug this manually when needed

# Pass controller to agent
agent = Agent(
	task=task, llm=llm, controller=controller, save_conversation_path=conversation_dir, generate_gif='agent_actions.gif'
)


async def print_step_info(agent_obj):
	"""Hook function to print detailed step information"""
	print('\n' + '=' * 80)
	print('🔍 STEP DEBUG INFO')
	print('=' * 80)

	# Get current step information
	step_num = agent_obj.state.n_steps
	print(f'📍 Current Step: {step_num}')

	# Print extracted content from this step
	if hasattr(agent_obj.state, 'history') and agent_obj.state.history.history:
		latest_history = agent_obj.state.history.history[-1] if agent_obj.state.history.history else None
		if latest_history and latest_history.result:
			print('\n🔗 EXTRACTED CONTENT:')
			for i, result in enumerate(latest_history.result):
				if result.extracted_content:
					print(f'  Result {i + 1}: {result.extracted_content[:200]}...')

		# Print model output (the agent's reasoning)
		if latest_history and latest_history.model_output:
			print('\n🧠 AGENT REASONING:')
			print(f'  Evaluation: {latest_history.model_output.current_state.evaluation_previous_goal}')
			print(f'  Memory: {latest_history.model_output.current_state.memory}')
			print(f'  Next Goal: {latest_history.model_output.current_state.next_goal}')

	print('=' * 80 + '\n')


async def main():
	# Setup Gmail authentication first
	gmail_ready = await setup_gmail()
	if not gmail_ready:
		print('⚠️  Continuing without Gmail - manual 2FA codes will be needed')

	# Show registered actions for debugging
	print(f'\n🔧 Controller has {len(controller.registry.registry.actions)} total actions registered')

	# Run the agent with step-by-step debugging
	print('🚀 Starting agent with debug logging enabled...')
	print('💡 Check the conversation_logs/ directory for saved prompts and responses')

	history = await agent.run(
		max_steps=15,  # Increased to allow for 2FA steps
		on_step_start=print_step_info,  # Print debug info before each step
	)

	# Print comprehensive results
	print('\n' + '=' * 100)
	print('📊 FINAL RESULTS ANALYSIS')
	print('=' * 100)

	# Token usage
	print(f'💰 Token Usage: {history.usage}')

	# All extracted content throughout the session
	all_extracted_content = history.extracted_content()
	print(f'\n📄 Total Extracted Content Items: {len(all_extracted_content)}')
	for i, content in enumerate(all_extracted_content):
		print(f'  Content {i + 1}: {content[:100]}...' if len(content) > 100 else f'  Content {i + 1}: {content}')

	# All URLs visited
	urls_visited = history.urls()
	print(f'\n🌐 URLs Visited: {len(urls_visited)}')
	for i, url in enumerate(urls_visited):
		print(f'  {i + 1}. {url}')

	# All actions taken
	action_names = history.action_names()
	print(f'\n⚡ Actions Taken: {action_names}')

	# Model outputs (agent's reasoning at each step)
	model_outputs = history.model_outputs()
	print(f'\n🤖 Agent Reasoning Steps: {len(model_outputs)}')
	for i, output in enumerate(model_outputs):
		print(f'  Step {i + 1}:')
		print(f'    Goal: {output.current_state.next_goal}')
		print(f'    Memory: {output.current_state.memory}')

	# Save detailed history to JSON for further analysis
	history_file = Path('./agent_history_detailed.json')
	with open(history_file, 'w') as f:
		history_data = {
			'task': task,
			'total_steps': len(history.history),
			'urls_visited': urls_visited,
			'actions_taken': action_names,
			'extracted_content': all_extracted_content,
			'token_usage': history.usage.model_dump() if history.usage else None,
			'success': history.is_successful(),
			'detailed_history': [
				{
					'step': i + 1,
					'url': h.state.url,
					'model_output': {
						'goal': h.model_output.current_state.next_goal if h.model_output else None,
						'memory': h.model_output.current_state.memory if h.model_output else None,
						'evaluation': h.model_output.current_state.evaluation_previous_goal if h.model_output else None,
						'actions': [action.model_dump(exclude_none=True) for action in h.model_output.action]
						if h.model_output
						else [],
					},
					'results': [r.model_dump(exclude_none=True) for r in h.result] if h.result else [],
				}
				for i, h in enumerate(history.history)
			],
		}
		json.dump(history_data, f, indent=2, default=str)

	print(f'\n💾 Detailed history saved to: {history_file}')
	print(f'💬 Conversation logs saved to: {conversation_dir}/')
	print('🎬 Action GIF saved to: agent_actions.gif')

	print('\n' + '=' * 100)
	print('✅ Analysis complete! Check the files above for detailed debugging information.')
	print('=' * 100)


if __name__ == '__main__':
	asyncio.run(main())
