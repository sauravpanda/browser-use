"""
Task Enhancement Example

This example demonstrates how the browser-use agent automatically enhances task descriptions
to make them more specific, actionable, and clear for browser automation.

The task enhancement feature:
1. Analyzes your original task description
2. Uses an LLM to make it more specific and actionable
3. Adds context, constraints, and clear steps
4. Provides better guidance for the automation agent

Key features demonstrated:
- Automatic task enhancement during agent initialization
- Comparing original vs enhanced tasks
- Enabling/disabling task enhancement
- Different types of tasks and their enhancements
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession

llm = ChatOpenAI(model='gpt-4o')


async def demonstrate_comprehensive_enhancement():
	"""Comprehensive demo showing task enhancement with comparison and execution."""

	print('🧠 Task Enhancement Comprehensive Demo')
	print('=' * 45)

	# Test cases that show clear enhancement benefits
	test_tasks = ['Find iPhone prices', 'Book flight to Paris', 'Search for laptop deals under $1000']

	async with BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,  # Start headless for comparison
			window_size={'width': 1280, 'height': 1000},
		)
	) as browser_session:
		print('📊 Comparing Original vs Enhanced Tasks:')
		print('-' * 45)

		enhanced_tasks = []

		for i, original_task in enumerate(test_tasks, 1):
			print(f"\n{i}. Original: '{original_task}'")

			# Create agent with enhancement
			agent = Agent(
				task=original_task,
				llm=llm,
				browser_session=browser_session,
				enhance_task=True,
			)

			enhanced_task = agent.task
			enhanced_tasks.append((original_task, enhanced_task, agent))

			print(f"   Enhanced: '{enhanced_task}'")

			# Show improvement metrics
			improvement = len(enhanced_task) - len(original_task)
			print(f'   📏 Added {improvement} characters for clarity')

		# Execute one enhanced task to show practical benefit
		print('\n🚀 Executing Enhanced Task Demo')
		print('-' * 45)

		# Use the first task for execution demo
		original_task, enhanced_task, agent = enhanced_tasks[0]

		print(f"📝 Executing: '{enhanced_task}'")
		print('🌐 Opening browser to demonstrate...')

		# Switch to visible browser for execution
		await browser_session.close()

	# Create new visible browser session for execution
	async with BrowserSession(
		browser_profile=BrowserProfile(
			headless=False,  # Show browser for execution
			window_size={'width': 1280, 'height': 1000},
		)
	) as browser_session:
		# Create agent for execution
		execution_agent = Agent(
			task=original_task,
			llm=llm,
			browser_session=browser_session,
			enhance_task=True,
		)

		try:
			print('🎯 Running enhanced task automation...')
			history = await execution_agent.run(max_steps=5)

			if history.is_successful:
				print('✅ Task completed successfully!')
				print(f'📋 Final result: {history.final_result()}')
			else:
				print('⏸️ Task demonstration completed')

		except Exception as e:
			print(f'⚠️ Demo stopped: {e}')

		print('\n💡 Key Benefits Demonstrated:')
		print('• Enhanced tasks provide clearer guidance to the agent')
		print('• More specific instructions lead to better automation results')
		print('• Task enhancement happens automatically - no extra work needed')
		print('• Works with any browser automation task')


async def main():
	"""Main function to run the comprehensive demonstration."""

	print('🌟 Browser-Use Task Enhancement Feature Demo')
	print('=' * 50)
	print('This demo shows how the browser-use agent automatically enhances')
	print('task descriptions to make them more specific and actionable.\n')

	try:
		# Run the comprehensive demonstration
		await demonstrate_comprehensive_enhancement()

		print('\n🎉 Demo completed successfully!')

	except Exception as e:
		print(f'\n❌ Demo failed: {e}')


if __name__ == '__main__':
	asyncio.run(main())
