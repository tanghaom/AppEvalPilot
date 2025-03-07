import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from metagpt.logs import logger


def draw_and_save_boxplot(excel_path, save_dir):
    """
    从Excel读取数据，绘制human_score和agent_score的箱线图，并保存到指定目录。

    Args:
        excel_path (str): Excel文件路径。
        save_dir (str): 图片保存目录。
    """
    try:
        # 设置Seaborn主题
        sns.set_theme(style="ticks", palette="pastel")

        # 读取Excel文件
        df = pd.read_excel(excel_path)

        # 过滤掉指定版本的数据
        df = df[df["执行版本"] != "cOd6c7cd"]
        df = df[df["执行版本"] != "071d8d8e"]
        df = df[df["执行版本"] != "jy-0e649f88"]

        # 创建一个新列 'score_type' 来区分 human_score 和 agent_score
        df_melted = pd.melt(
            df,
            value_vars=["human_score", "agent_avail_score_test(去除不可靠用例)"],
            var_name="score_type",
            value_name="score",
            id_vars="执行版本",
        )

        # 创建箱线图, 调整figsize和箱体透明度
        plt.figure(figsize=(12, 8))  # 调整宽高比
        ax = sns.boxplot(
            x="执行版本", y="score", hue="score_type", palette=["m", "g"], data=df_melted, boxprops=dict(alpha=0.7)
        )  # 增加透明度

        # 获取图例对象
        handles, labels = ax.get_legend_handles_labels()

        # 修改图例标签
        labels = ["human_score", "agent_avail_score"]
        ax.legend(handles, labels)

        # 去除箱线图的边框
        sns.despine(offset=10, trim=True)

        # 设置图表标题和标签
        plt.title("Distribution of Human Score vs Agent Score by Version")
        plt.xlabel("Version")
        plt.ylabel("Score")

        # 调整布局
        plt.tight_layout()

        # 定义保存图片的路径
        save_path = os.path.join(save_dir, "score_comparison_boxplot.png")

        # 保存图片
        plt.savefig(save_path)
        logger.info(f"箱线图已保存到: {save_path}")

        # 展示图表
        plt.show()

    except FileNotFoundError:
        logger.error(f"文件未找到: {excel_path}")
    except Exception as e:
        logger.error(f"绘制箱线图时发生错误: {e}")


def draw_jointplot(excel_path: str, save_dir: str):
    """
    绘制散点图以显示human_score和agent_avail_score之间的相关性，横轴为版本。

    Args:
        excel_path (str): Excel文件路径。
        save_dir (str): 保存图片的目录。
    """
    try:
        sns.set_theme(style="ticks")
        # 读取Excel文件
        df = pd.read_excel(excel_path)

        # 过滤掉指定版本的数据
        df = df[df["执行版本"] != "cOd6c7cd"]
        df = df[df["执行版本"] != "071d8d8e"]
        df = df[df["执行版本"] != "jy-0e649f88"]

        # 准备数据
        df.rename(columns={"agent_avail_score_test(去除不可靠用例)": "agent_score"}, inplace=True)

        # 获取所有唯一的执行版本
        versions = df["执行版本"].unique()

        # 循环遍历每个版本
        for version in versions:
            # 过滤当前版本的数据
            version_data = df[df["执行版本"] == version]

            # 提取human_score和agent_score
            human_scores = version_data["human_score"].tolist()
            agent_scores = version_data["agent_score"].tolist()

            # 检查列表是否为空或长度不匹配
            if not human_scores or not agent_scores or len(human_scores) != len(agent_scores):
                logger.warning(f"版本 {version} 的数据不足或human_score和agent_score长度不匹配，跳过。")
                continue

            # 计算皮尔逊相关系数
            correlation = np.corrcoef(human_scores, agent_scores)[0, 1]
            logger.info(f"版本 {version} 的 Pearson correlation coefficient: {correlation:.2f}")

            # 绘制散点图
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=human_scores, y=agent_scores)
            plt.title(f"版本: {version}, Correlation: {correlation:.2f}")
            plt.xlabel("Human Score")
            plt.ylabel("Agent Score")
            plt.tight_layout()

            # 保存图片
            save_path = os.path.join(save_dir, f"{version}_correlation_scatter.png")
            plt.savefig(save_path)
            logger.info(f"版本 {version} 的散点图已保存到: {save_path}")

            # 展示图表
            plt.show()
            plt.close()  # 关闭图形，防止内存泄漏

    except FileNotFoundError:
        logger.error(f"文件未找到: {excel_path}")
    except Exception as e:
        logger.error(f"绘制散点图时发生错误: {e}")


if __name__ == "__main__":
    # 定义Excel文件路径和保存图片的路径
    excel_path = "data/自动测试用例.xlsx"
    save_dir = "data/draw"

    # 调用函数绘制并保存箱线图
    # draw_and_save_boxplot(excel_path, save_dir)
    draw_jointplot(excel_path, save_dir)
