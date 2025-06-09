import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class WeightTracker:
    # 常量定义
    MEAL_FIELDS = ["breakfast", "lunch", "dinner"]
    FIELD_NAMES = ["weight", "realizing_breakfast", "done_breakfast",
                   "realizing_lunch", "done_lunch", "realizing_dinner",
                   "done_dinner", "realizing_exercise", "done_exercise",
                   "exhaustion_level"]

    # 目标体重线
    WEIGHT_LINES = {
        "标准线": 106.9,
        "偏重线": 138.7,
        "过重线": 161.8
    }

    # 雷达图标签映射
    RADAR_FIELD_MAP = {
        "早饭意识到": "realizing_breakfast",
        "早饭做到": "done_breakfast",
        "午饭意识到": "realizing_lunch",
        "午饭做到": "done_lunch",
        "晚饭意识到": "realizing_dinner",
        "晚饭做到": "done_dinner",
        "意识到运动": "realizing_exercise",
        "专门做了运动": "done_exercise"
    }

    def __init__(self, data_file="weight.log"):
        self.data_file = data_file
        self.data_all = {}
        self.load_data()
        # 确保图片保存目录存在
        self.img_save_dir = "save_img"
        os.makedirs(self.img_save_dir, exist_ok=True)

    def load_data(self):
        try:
            with open(self.data_file, 'r') as file:
                self.data_all = json.load(file)
                print(f"日志已加载，共 {len(self.data_all)} 条记录")
        except FileNotFoundError:
            print("未找到数据文件，将创建新文件")
            self.data_all = {}
        except json.JSONDecodeError:
            print("数据文件格式错误，将创建新文件")
            self.data_all = {}

    def save_data(self):
        with open(self.data_file, 'w') as file:
            json.dump(self.data_all, file, indent=2)
        print(f"数据已保存到 {self.data_file}")

    def process_input(self, prompt, default=None, choices=None):
        """通用输入处理函数"""
        while True:
            value = input(prompt).strip().lower()
            if not value and default is not None:
                return default
            if choices and value not in choices:
                print(f"请输入有效的选项: {', '.join(choices)}")
                continue
            return value

    def input_meal_data(self, meal_type):
        """输入特定餐食的数据"""
        data = {}
        data[f"realizing_{meal_type}"] = self.process_input(
            f"昨日{meal_type}是否意识到(y/n，默认为n): ", "n", ["y", "n"]
        )
        data[f"done_{meal_type}"] = self.process_input(
            f"昨日{meal_type}是否做到(y/n，默认为n): ", "n", ["y", "n"]
        )
        return data

    def input_data(self):
        """输入体重数据"""
        while True:
            date_str = self.process_input("请输入当前记录时间(格式:YYYYMMDD_HHMM): ")

            # 验证日期格式
            try:
                datetime.strptime(date_str, "%Y%m%d_%H%M")
            except ValueError:
                print("日期格式错误，请使用 YYYYMMDD_HHMM 格式")
                continue

            # 检查是否已存在
            if date_str in self.data_all:
                overwrite = self.process_input("已有同一时刻数据，是否覆盖？(y/n): ", "n", ["y", "n"])
                if overwrite != "y":
                    continue

            # 创建新记录
            record = {field: None for field in self.FIELD_NAMES}

            # 输入体重
            try:
                record["weight"] = float(self.process_input("今早体重(斤): "))
            except ValueError:
                print("请输入有效的体重数值")
                continue

            # 输入餐食数据
            for meal in self.MEAL_FIELDS:
                meal_data = self.input_meal_data(meal)
                record.update(meal_data)

            # 输入运动数据
            record["realizing_exercise"] = self.process_input(
                "昨日是否意识到运动(y/n，默认为n): ", "n", ["y", "n"]
            )
            record["done_exercise"] = self.process_input(
                "昨日是否专门运动了(y/n，默认为n): ", "n", ["y", "n"]
            )

            # 输入疲劳程度
            exhaustion = self.process_input(
                "昨日全天消耗劳累程度评估(多了o/适中m/少了l，默认为m): ", "m", ["o", "m", "l"]
            )
            record["exhaustion_level"] = {"o": 1, "m": 0, "l": -1}[exhaustion]

            # 转换布尔值为数字
            for key in record:
                if isinstance(record[key], str):
                    record[key] = 1 if record[key] == "y" else 0

            # 保存记录
            self.data_all[date_str] = record
            print(f"{date_str} 的数据已记录")

            # 检查是否继续
            if self.process_input("继续请输入1，退出请直接回车: ") != "1":
                self.save_data()
                return

    def delete_data(self):
        """删除数据"""
        while True:
            date_to_delete = input("请输入想删除数据的日期(或输入q退出): ").strip()
            if date_to_delete.lower() == "q":
                return

            if date_to_delete in self.data_all:
                del self.data_all[date_to_delete]
                print(f"{date_to_delete} 的数据已删除")
                self.save_data()
            else:
                print("无此数据！")

    def plot_weight_chart(self):
        """绘制体重变化折线图并保存"""
        if not self.data_all:
            print("没有可用数据")
            return

        # 准备数据
        sorted_dates = sorted(self.data_all.keys())
        weights = [self.data_all[date]["weight"] for date in sorted_dates]

        # 格式化日期显示
        display_dates = [f"{date[0:4]}_{date[4:8]}" for date in sorted_dates]

        # 创建图表
        plt.figure(figsize=(12, 6))
        plt.plot(display_dates, weights, 'o-', linewidth=2, markersize=8, label='体重')

        # 添加数据点标签
        for i, weight in enumerate(weights):
            plt.annotate(f'{weight:.1f}',
                         (display_dates[i], weight),
                         xytext=(0, 10),
                         textcoords='offset points',
                         ha='center')

        # 设置图表属性
        plt.xlabel("日期")
        plt.xticks(rotation=45)
        plt.ylabel("体重 (斤)", rotation=0, labelpad=20)
        plt.title("体重变化趋势")

        # 设置Y轴范围
        min_weight, max_weight = min(weights), max(weights)
        y_padding = (max_weight - min_weight) * 0.2
        plt.ylim(min_weight - y_padding, max_weight + y_padding)

        # 添加目标线 - 只在体重接近时显示
        visible_lines = []
        for label, weight in self.WEIGHT_LINES.items():
            # 判断目标线是否在当前体重范围内
            if min_weight - 10 <= weight <= max_weight + 10:
                color = 'r' if "过重" in label else 'g'
                plt.axhline(y=weight, color=color, linestyle='--', alpha=0.7)
                # 根据位置调整文本位置
                if weight > max_weight:
                    text_y = max_weight - 1
                    va = 'top'
                elif weight < min_weight:
                    text_y = min_weight + 1
                    va = 'bottom'
                else:
                    text_y = weight + 0.2
                    va = 'bottom'

                plt.text(0.01, text_y, f"{label}: {weight}",
                         color=color,
                         transform=plt.gca().get_yaxis_transform(),
                         verticalalignment=va)
                visible_lines.append(label)

        # 如果没有显示任何目标线，添加图例说明
        if not visible_lines:
            plt.text(0.01, 0.95, "目标体重线在当前范围外未显示",
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.7))

        # 计算并添加95%置信区间
        if len(weights) > 1:
            try:
                # 计算均值和标准误
                mean = np.mean(weights)
                std = np.std(weights, ddof=1)  # 样本标准差
                n = len(weights)
                se = std / np.sqrt(n)

                # 计算95%置信区间
                t_value = stats.t.ppf(0.975, df=n - 1)
                margin = t_value * se
                lower_bound = mean - margin
                upper_bound = mean + margin

                # 绘制置信区间
                plt.axhline(mean, color='blue', linestyle='-', alpha=0.7, label='均值')
                plt.fill_between(
                    display_dates,
                    lower_bound,
                    upper_bound,
                    color='blue',
                    alpha=0.2,
                    label='95%CI'
                )

                # 添加置信区间标签
                plt.text(
                    0.5, mean + margin + 0.5,
                    f"95%CI: {lower_bound:.1f} - {upper_bound:.1f}",
                    color='blue', ha='center', transform=plt.gca().get_yaxis_transform()
                )
            except Exception as e:
                print(f"计算置信区间时出错: {e}")

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.img_save_dir, f"{timestamp}_weight_chart.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"体重图表已保存为: {filename}")

        plt.show()

    def plot_radar_chart(self):
        """绘制行为分析雷达图并保存"""
        if not self.data_all:
            print("没有可用数据")
            return

        # 统计数据 - 修复None值问题
        stats = {label: 0 for label in self.RADAR_FIELD_MAP.keys()}

        for record in self.data_all.values():
            for label, field in self.RADAR_FIELD_MAP.items():
                # 处理可能的None值
                value = record.get(field)
                if value is None:
                    # 对于新数据，None可能表示未输入，视为0
                    value = 0
                elif isinstance(value, str):
                    # 对于旧数据，处理字符串类型的值
                    value = 1 if value.lower() == "y" else 0
                stats[label] += value

        values = list(stats.values())
        total_records = len(self.data_all)

        # 转换为百分比
        if total_records > 0:
            values = [v / total_records * 100 for v in values]
        else:
            values = [0] * len(values)  # 防止除零错误

        # 雷达图标签
        labels = list(self.RADAR_FIELD_MAP.keys())

        # 雷达图设置
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        # 创建雷达图
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)

        # 绘制数据
        ax.plot(angles, values, 'o-', linewidth=2, markersize=8)
        ax.fill(angles, values, alpha=0.25, color='#1f77b4')

        # 设置标签
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # 添加数据点标签
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            ax.text(angle, value + 5, f"{value:.1f}%",
                    ha='center', va='center', fontsize=10)

        # 设置网格和标题
        ax.set_rlabel_position(0)
        plt.yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"],
                   color="grey", size=10)
        plt.ylim(0, 110)
        plt.title("健康行为分析雷达图", size=15, pad=20)

        # 添加统计信息
        plt.figtext(0.5, 0.95,
                    f"\n基于 {total_records} 条记录的行为分析",
                    ha='center', fontsize=9)

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.img_save_dir, f"{timestamp}_radar_chart.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"雷达图表已保存为: {filename}")

        plt.show()


if __name__ == "__main__":
    tracker = WeightTracker()

    while True:
        print("\n" + "=" * 40)
        print("体重跟踪分析系统")
        print("=" * 40)
        print("1. 输入数据")
        print("2. 删除数据")
        print("3. 查看体重图表")
        print("4. 查看行为分析雷达图")
        print("5. 退出")

        choice = input("请选择操作 (1-5): ").strip()

        if choice == "1":
            tracker.input_data()
        elif choice == "2":
            tracker.delete_data()
        elif choice == "3":
            tracker.plot_weight_chart()
        elif choice == "4":
            tracker.plot_radar_chart()
        elif choice == "5":
            print("再见！")
            break
        else:
            print("无效选择，请重新输入")