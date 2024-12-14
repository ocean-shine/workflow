import base64

class PromptVideo:
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        """
        构建系统的提示信息，系统提示通常用于为模型提供上下文，指导模型行为。
        """
        system_prompt = """
            You are an expert teacher capable of summarizing visual and transcribed content.
            You will receive image frames and text from a video. Please generate a concise and accurate summary based on the provided information.
        """

        return system_prompt

    def build_user_prompt(self, context: str) -> dict:
        """
        构建用户的提示信息，基于用户的查询和上下文，生成适合的提示内容。
        :param context: 包含用户查询和上下文的字符串。
        :return: 构建的用户提示信息。
        """
        user_prompt = {
            "type": "text",
            "text": context
        }
        return user_prompt

    import base64

    def process_images_and_text(self, retrieved_texts: list, image_paths: list, top_k_text_indices: list, image_per_text_indices: list) -> dict:
        """
        处理图像和文本的组合，将与检索到的文本块相关的图像和文本一起返回。
        :param retrieved_texts: 检索到的文本块
        :param image_paths: 图像路径列表
        :param top_k_text_indices: 每个图像块的前 k 个文本索引
        :param top_k_image_indices: 每个文本块的前 k 个图像索引
        :return: 结合文本和图像的结果
        """
        # 构建用户消息内容
        user_message_content = []

        # 遍历每个文本块及其相关的图像
        for idx in range(len(top_k_text_indices)):
            # 获取与该文本块相关的图像索引
            top_image_indices_for_text = image_per_text_indices[idx]  # 获取每个文本块的前 k 张图像的索引
            text = retrieved_texts[idx]
            base64frames_for_text = []

            user_message_content.append({
                "type": "text",
                "text": text  # 当前文本块
            })
            
            # 读取与该文本块相关的图像（base64 编码）
            for image_idx in top_image_indices_for_text:
                image_path = image_paths[image_idx]  # 获取图像路径
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')  # 图像转为 base64 编码
                    base64frames_for_text.append(base64_image)

            # 每个文本块的前 k 张相关图像
            for image_data in base64frames_for_text:
                user_message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "high"}
                })

           
        return user_message_content

            
class PromptText:
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        """
        构建系统的提示信息，系统提示通常用于为模型提供上下文，指导模型行为。
        """
        system_prompt = """
            You are an expert who can understand and process textual content. Please generate a summary or answer related questions based on the provided text.
        """
        return system_prompt

    def build_user_prompt(self, context: str) -> dict:
        """
        构建用户的提示信息，基于用户的查询和上下文，生成适合的提示内容。
        :param context: 包含用户查询和上下文的字符串。
        :return: 构建的用户提示信息。
        """
        user_prompt = {
            "type": "text",
            "text": context
        }
        return user_prompt


class PromptPDF:
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        """
        构建系统的提示信息，系统提示通常用于为模型提供上下文，指导模型行为。
        """
        system_prompt = """
            You are an expert who can understand and process the content of PDF documents. Please answer relevant questions or summarize the document based on the provided PDF text or content.
        """
        return system_prompt

    def build_user_prompt(self, context: str) -> dict:
        """
        构建用户的提示信息，基于用户的查询和上下文，生成适合的提示内容。
        :param context: 包含用户查询和上下文的字符串。
        :return: 构建的用户提示信息。
        """
        user_prompt = {
            "type": "text",
            "text": context
        }
        return user_prompt


class PromptAudio:
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        """
        构建系统的提示信息，系统提示通常用于为模型提供上下文，指导模型行为。
        """
        system_prompt = """
            You are an expert who can analyze and understand the transcribed text of audio content. Please generate a summary or answer relevant questions based on the provided audio transcript.
        """

        return system_prompt

    def build_user_prompt(self, context: str) -> dict:
        """
        构建用户的提示信息，基于用户的查询和上下文，生成适合的提示内容。
        :param context: 包含用户查询和上下文的字符串。
        :return: 构建的用户提示信息。
        """
        user_prompt = {
            "type": "text",
            "text": context
        }
        return user_prompt
