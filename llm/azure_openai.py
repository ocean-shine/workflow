# llm/azure_openai.py
import openai
from openai import AzureOpenAI
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMText:
    def __init__(self, model_name: str, api_key: str, api_base: str, api_version: str):
        """
        初始化 LLMText 类，用于与 Azure OpenAI 进行通信并处理文本内容。

        :param model_name: Azure OpenAI 模型名称
        :param api_key: Azure API 密钥
        :param api_base: Azure API 基础 URL
        :param api_version: Azure API 版本
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

        # 初始化 Azure OpenAI 客户端
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.api_base,
            api_version=self.api_version
        )

    def generate_text(self, system_prompt: dict, user_prompt: dict) -> dict:
        """
        调用 Azure OpenAI API 生成文本模型的响应，支持文本输入。

        :param system_prompt: 系统级提示（字典形式）
        :param user_prompt: 用户查询提示（字典形式）
        :return: LLM 返回的响应
        """
        try:
            # 构建请求消息
            messages = [
                {"role": "system", "content": system_prompt['content']},
                {"role": "user", "content": user_prompt['text']}
            ]

            # 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,  # 设置生成的最大 token 数量
                temperature=0.3,  # 设置生成的随机性
            )

            # 处理响应并返回
            return response['choices'][0]['message']['content']
        
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            return {"error": f"Error generating response: {e}"}


class LLMPDF:
    def __init__(self, model_name: str, api_key: str, api_base: str, api_version: str):
        """
        初始化 LLMPDF 类，用于与 Azure OpenAI 进行通信并处理 PDF 内容。

        :param model_name: Azure OpenAI 模型名称
        :param api_key: Azure API 密钥
        :param api_base: Azure API 基础 URL
        :param api_version: Azure API 版本
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

        # 初始化 Azure OpenAI 客户端
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.api_base,
            api_version=self.api_version
        )

    def generate_pdf(self, system_prompt: dict, user_prompt: dict) -> dict:
        """
        调用 Azure OpenAI API 生成PDF任务的响应，支持PDF文件的文本内容。

        :param system_prompt: 系统级提示（字典形式）
        :param user_prompt: 用户查询提示（字典形式）
        :return: LLM 返回的响应
        """
        try:
            # 构建请求消息
            messages = [
                {"role": "system", "content": system_prompt['content']},
                {"role": "user", "content": user_prompt['text']}
            ]

            # 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,  # 设置生成的最大 token 数量
                temperature=0.3,  # 设置生成的随机性
            )

            # 处理响应并返回
            return response['choices'][0]['message']['content']
        
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            return {"error": f"Error generating response: {e}"}


class LLMAudio:
    def __init__(self, model_name: str, api_key: str, api_base: str, api_version: str):
        """
        初始化 LLMAudio 类，用于与 Azure OpenAI 进行通信并处理音频内容。

        :param model_name: Azure OpenAI 模型名称
        :param api_key: Azure API 密钥
        :param api_base: Azure API 基础 URL
        :param api_version: Azure API 版本
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

        # 初始化 Azure OpenAI 客户端
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.api_base,
            api_version=self.api_version
        )

    def generate_audio(self, system_prompt: dict, user_prompt: dict) -> dict:
        """
        调用 Azure OpenAI API 生成音频任务的响应，支持音频文件的文本内容。

        :param system_prompt: 系统级提示（字典形式）
        :param user_prompt: 用户查询提示（字典形式）
        :return: LLM 返回的响应
        """
        try:
            # 构建请求消息
            messages = [
                {"role": "system", "content": system_prompt['content']},
                {"role": "user", "content": user_prompt['text']}
            ]

            # 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,  # 设置生成的最大 token 数量
                temperature=0.3,  # 设置生成的随机性
            )

            # 处理响应并返回
            return response['choices'][0]['message']['content']
        
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            return {"error": f"Error generating response: {e}"}


class LLMVideo:
    def __init__(self, model_name: str, api_key: str, api_base: str, api_version: str):
        """
        初始化 LLMVideo 类，用于与 Azure OpenAI 进行通信并处理视频内容。

        :param model_name: Azure OpenAI 模型名称
        :param api_key: Azure API 密钥
        :param api_base: Azure API 基础 URL
        :param api_version: Azure API 版本
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        openai.api_type = "azure"
        # 初始化 Azure OpenAI 客户端
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.api_base,
            api_version=self.api_version
        )

    def generate_video(self, system_prompt: dict, user_prompt: dict) -> dict:
        """
        调用 Azure OpenAI API 生成模型的响应，支持图像和文本输入的混合。

        :param system_prompt: 系统级提示（字典形式）
        :param user_prompt: 用户查询提示（字典形式）
        :return: LLM 返回的响应
        """
        try:
            # 构建请求消息
            messages = [
                {"role": "system", "content": system_prompt['content']},
                {"role": "user", "content": user_prompt['text']}
            ]

            # 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,  # 设置生成的最大 token 数量
                temperature=0.3,  # 设置生成的随机性
            )

            # 处理响应并返回
            return response.choices[0].message.content

        
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            return {"error": f"Error generating response: {e}"}
