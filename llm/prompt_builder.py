# llm/prompt_builder.py

def build_system_prompt() -> str:
    """
    构建系统提示，定义 AI 角色和行为。
    """
    return """
    You are an expert teacher that summarizes visual and transcribed content.
    """

def build_user_prompt(user_message_content: list) -> list:
    """
    构建用户提示，包含文本和图像信息。

    :param user_message_content: 包含文本和图像的内容列表。
    :return: 格式化后的消息列表。
    """
    return [
        {"role": "user", "content": user_message_content}
    ]
