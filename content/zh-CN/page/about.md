---
title: "关于我"
date: 2025-07-02
toc: false
---

你好，欢迎来到我的小站 👋

我是一个在意大利读博的普通中国留学生，现在在米兰理工大学做关于遥感、地理信息和 AI 的研究。平时跟卫星影像、高分辨率地图、地表分类这些东西打交道，也写代码、跑模型、画图表，偶尔熬夜写论文做PPT😅。

除了科研，我也喜欢整理笔记、搭建网站、做点开发项目。这个博客，就是我记录学习、分享想法、输出内容的地方。

你在这里会看到一些内容，比如：
- 遥感与地理数据处理的教程 / 工具总结  
- 自己做研究时踩过的坑 & 实用技巧分享  
- 跟学术有关的小碎片，比如论文、PPT 或会议感想  
- 还有一些乱七八糟的想法 & 研究经历  

本站内容大多是用 Markdown 写的，放在 GitHub 上管理，用 Cloudflare Pages 自动部署上线（技术人最爱组合🤓）。

如果你碰巧也在做遥感 / GIS / AI / 博客写作，或者只是偶然路过想说句话，欢迎随时联系我！

---

📍 目前所在：意大利 米兰  
📫 联系方式：[ele.qiong@gmail.com]   
🔗 GitHub: [github.com/ictar](https://github.com/ictar)


## 我的足迹

<div id="map" style="height: 600px; margin-top: 2rem;"></div>

<script>
    document.addEventListener("DOMContentLoaded", function () {
        // 基础地图初始化
        const map = L.map('map').setView([45.4782, 9.2276], 8);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 28,
        }).addTo(map);

        // 当前语言判断
        const lang = document.documentElement.lang.includes('zh') ? 'zh-CN' : 'en';

        // 自定义图标（按类型区分）
        const iconMap = {
            study: L.AwesomeMarkers.icon({
                icon: 'graduation-cap',
                prefix: 'fa',
                markerColor: 'blue'
            }),
            work: L.AwesomeMarkers.icon({
                icon: 'briefcase',
                prefix: 'fa',
                markerColor: 'red'
            }),
            travel: L.AwesomeMarkers.icon({
                icon: 'route',
                prefix: 'fa',
                markerColor: 'green'
            })
        };

        // 加载对应语言的 JSON 文件
        fetch(`/data/places.${lang}.json`)
        .then(res => res.json())
        .then(data => {
            data.forEach(place => {
            const marker = L.marker(place.coords, { icon: iconMap[place.type] });
            const popupHtml = `
                <div style="min-width:180px;padding:8px 0;">
                    <div style="font-weight:bold;font-size:16px;margin-bottom:4px;">
                        <i class="fa fa-${place.type === 'study' ? 'graduation-cap' : place.type === 'work' ? 'briefcase' : 'plane'}" style="margin-right:6px;color:#4A90E2;"></i>
                        ${place.title}
                    </div>
                    <div style="color:#555;">${place.desc}</div>
                    <div style="font-size:12px;color:#888;margin-top:4px;">${place.years}</div>
                </div>
            `;
            marker.bindPopup(popupHtml);
            marker.addTo(map);
            });
        });

        // 图例
        const legend = L.control({ position: 'bottomright' });

        legend.onAdd = function () {
        const div = L.DomUtil.create('div', 'custom-legend');
        div.innerHTML = `
            <div class="legend-title">📍 我的经历</div>
            <div class="legend-item">
                <i class="fa fa-graduation-cap" style="color:#4A90E2;margin-right:6px;"></i> 学习
            </div>
            <div class="legend-item">
                <i class="fa fa-briefcase" style="color:#D0021B;margin-right:6px;"></i> 工作
            </div>
            <div class="legend-item">
                <i class="fa fa-route" style="color:#7ED321;margin-right:6px;"></i> 旅行
            </div>
        `;
        return div;
        };

        legend.addTo(map);

    })
</script>
<style>
    .custom-legend {
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  padding: 12px 16px;
  border-radius: 12px;
  font-family: "Helvetica Neue", sans-serif;
  font-size: 14px;
  color: #333;
  line-height: 1.6;
  max-width: 200px;
}

.custom-legend .legend-title {
  font-weight: bold;
  margin-bottom: 8px;
  font-size: 15px;
  color: #222;
}

.custom-legend .legend-item {
  display: flex;
  align-items: center;
  margin-bottom: 6px;
}

.custom-legend .legend-icon {
  width: 12px;
  height: 12px;
  display: inline-block;
  margin-right: 8px;
  border-radius: 3px;
}

</style>