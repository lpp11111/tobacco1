import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
# 替换为Windows系统自带微软雅黑
plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------
# 模块1：独立计算【中尺度YOLO识别病虫害】
# --------------------------
print("==================== 模块1：中尺度YOLO模型独立统计 ====================")
gdf_yolo = gpd.read_file("tobacco_plant.shp")
total_all = len(gdf_yolo)

# 打印全部类别分布
print("class_id 全部统计分布：")
print(gdf_yolo["class_id"].value_counts())

yolo_stat = gdf_yolo["class_id"].value_counts().sort_index()
yolo_healthy = yolo_stat.get(0, 0)
# 病害class_id=2
yolo_disease = yolo_stat.get(2, 0)
yolo_disease_ratio = round(yolo_disease / total_all, 3)

print(f"全域总识别烟株：{total_all} 株")
print(f"中尺度YOLO判定健康植株(class_id=0)：{yolo_healthy} 株")
print(f"中尺度YOLO识别病虫害植株(class_id=2)：{yolo_disease} 株")
print(f"YOLO识别病害占全部烟株比例：{yolo_disease_ratio}\n")

# 导出中尺度单独统计表
table_yolo_single = pd.DataFrame({
    "项目": ["总烟株数量", "健康植株数量", "病虫害植株数量", "病害植株占比"],
    "数值": [total_all, yolo_healthy, yolo_disease, yolo_disease_ratio]
})
table_yolo_single.to_excel("01_中尺度YOLO单独统计.xlsx", index=False)
print("中尺度独立统计表已保存：01_中尺度YOLO单独统计.xlsx\n")

# --------------------------
# 模块2：独立计算【NDVI胁迫判别】
# --------------------------
print("==================== 模块2：NDVI植被指数独立统计 ====================")
gdf_ndvi = gpd.read_file("tobacco_plant_ndvi.shp")
ndvi_stat = gdf_ndvi["disease"].value_counts()
ndvi_healthy = ndvi_stat.get("健康植株", 0)
ndvi_stress = ndvi_stat.get("疑似病害", 0)
ndvi_stress_ratio = round(ndvi_stress / total_all, 3)

print(f"NDVI统计总烟株：{total_all} 株")
print(f"NDVI判定健康植株：{ndvi_healthy} 株")
print(f"NDVI判定疑似胁迫病害植株：{ndvi_stress} 株")
print(f"NDVI胁迫植株占全部烟株比例：{ndvi_stress_ratio}\n")

# 导出NDVI单独统计表
table_ndvi_single = pd.DataFrame({
    "项目": ["总烟株数量", "健康植株数量", "NDVI胁迫异常植株数量", "胁迫植株占比"],
    "数值": [total_all, ndvi_healthy, ndvi_stress, ndvi_stress_ratio]
})
table_ndvi_single.to_excel("02_NDVI单独统计.xlsx", index=False)
print("NDVI独立统计表已保存：02_NDVI单独统计.xlsx\n")

# --------------------------
# 模块3：两套结果交叉对比（病害class_id=2）
# --------------------------
print("==================== 模块3：YOLO与NDVI交叉对比分析 ====================")
gdf_ndvi = gdf_ndvi.to_crs(gdf_yolo.crs)
gdf_merge = gdf_yolo.merge(gdf_ndvi[["geometry", "ndvi_mean", "disease"]], on="geometry", how="left")

both_abnormal = gdf_merge[(gdf_merge["class_id"] == 2) & (gdf_merge["disease"] == "疑似病害")]
only_yolo_abnormal = gdf_merge[(gdf_merge["class_id"] == 2) & (gdf_merge["disease"] == "健康植株")]
only_ndvi_abnormal = gdf_merge[(gdf_merge["class_id"] == 0) & (gdf_merge["disease"] == "疑似病害")]

print(f"两种方法同时判定异常（高可信度病株）：{len(both_abnormal)} 株")
print(f"仅中尺度YOLO判定病害（模型误检）：{len(only_yolo_abnormal)} 株")
print(f"仅NDVI判定胁迫（模型漏检）：{len(only_ndvi_abnormal)} 株\n")

# 导出对比汇总表
table_compare = pd.DataFrame({
    "判别方法": ["中尺度YOLO视觉识别", "NDVI植被指数判别"],
    "总烟株数": [total_all, total_all],
    "健康植株": [yolo_healthy, ndvi_healthy],
    "异常植株": [yolo_disease, ndvi_stress],
    "异常植株占比": [yolo_disease_ratio, ndvi_stress_ratio]
})
table_diff = pd.DataFrame({
    "差异类型": ["两者同步判定异常", "仅YOLO识别病害(误检)", "仅NDVI判定胁迫(漏检)"],
    "植株数量": [len(both_abnormal), len(only_yolo_abnormal), len(only_ndvi_abnormal)],
    "占总株比例": [
        round(len(both_abnormal)/total_all,3),
        round(len(only_yolo_abnormal)/total_all,3),
        round(len(only_ndvi_abnormal)/total_all,3)
    ]
})
with pd.ExcelWriter("03_两套方法对比汇总表.xlsx") as writer:
    table_compare.to_excel(writer, sheet_name="总量对比", index=False)
    table_diff.to_excel(writer, sheet_name="差异明细", index=False)
print("两套方法对比汇总表已保存：03_两套方法对比汇总表.xlsx")

# 生成左右分栏独立对比图，删除plt.show()避免报错
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
# 左图：中尺度YOLO识别分布
gdf_merge.plot(column="class_id", cmap="RdYlGn", ax=ax1, legend=True)
ax1.set_title("中尺度YOLO病虫害识别分布（0=健康，2=病害）")
ax1.set_axis_off()
# 右图：NDVI胁迫分布
gdf_merge.plot(column="ndvi_mean", cmap="Greens", ax=ax2, legend=True)
ax2.set_title("NDVI植被指数胁迫分布")
ax2.set_axis_off()
plt.tight_layout()
plt.savefig("两套方法独立对比图.png", dpi=300)
print("对比图片已保存至文件夹，无需弹窗显示")
