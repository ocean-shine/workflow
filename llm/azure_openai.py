from config import settings
from openai import AzureOpenAI

class AzureLLM:
    def __init__(self):
        self.api_key = settings.AZURE_API_KEY
        self.api_base = settings.AZURE_API_BASE
        self.api_version = "2024-08-01-preview"
        self.model = settings.MODEL_DEPLOYMENT

        # Initialize the Azure OpenAI client
        self.client = AzureOpenAI(
            api_key = self.api_key,
            azure_endpoint=self.api_base,
            api_version=self.api_version
        )

    def generate(self, query, context):
        """
        Generate a response using Azure OpenAI LLM.

        :param query: User's input query.
        :param context: Relevant context retrieved for the query.
        :return: Generated response from the LLM.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": " ".join(context)},
        ]

        try:
            # Use Azure OpenAI client to create chat completion
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            # Return the content of the first choice in the response
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error generating response: {e}")
