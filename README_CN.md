## ComfyUI-JanusProCaption

[English](README) | 简体中文

**ComfyUI-JanusProCaption** 是一款专注于生成图像描述的工具，由 Janus Pro 模型提供支持。为用户提供了一种智能且简便的方式来描述图像内容。不论是用在图生图流程，还是为 LoRA 模型训练生成描述文本，这款工具都能满足需求。

特别感谢 https://github.com/deepseek-ai/Janus 发布了功能强大的 Janus Pro 模型，以及 https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro 在 ComfyUI 中实现了使用 Janus Pro 进行图像生成和理解的方法。在他们的基础上，我结合自己的需求进行了改动。此项目专注于利用 Janus Pro 进行图像理解，包括单张图像的描述，以及为 LoRA 训练数据集制作字幕文件等功能。

### 已实现功能

1. **描述单张图像**，借助 Janus Pro 模型对单张图像进行详细描述。该功能允许用户上传图像，并通过提问生成相应的详细描述。<br>

2. **批量描述图像**，借助 Janus Pro 模型自动为指定目录中的所有图像生成描述文本。该功能会逐一处理每张图像，生成描述内容，并将描述保存为对应的 .txt 文件。<br>

### 联系我

- B站: [@黎黎原上咩](https://space.bilibili.com/449342345)
- YouTube: [@SweetValberry](https://www.youtube.com/@SweetValberry)
