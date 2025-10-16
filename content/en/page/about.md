---
title: "About Me"
date: 2025-07-02
---

👋 Hi, I'm a researcher, developer, and content creator.

I’m currently a research fellow at the **GeoLab, Politecnico di Milano**, where I work on high-resolution land cover mapping, Earth observation, and AI-based remote sensing analysis. I enjoy exploring how satellite data and geospatial models can help us better understand urban and environmental change.


This blog is my digital garden — a space for:
- Sharing **technical tutorials** on GIS, remote sensing, and web development
- Posting notes from **academic work**, especially validation methods and dataset benchmarking

You’ll find most of my content written in **Markdown**, versioned on **GitHub**, and deployed with ❤️ on **Cloudflare Pages**.

If you’d like to collaborate, chat, or just say hi, feel free to reach out!

---

📍 Currently in: Milan, Italy  
🌐 Languages: Chinese (native), English (academic), Italian (learning A2)  
📫 Contact: [ele.qiong@gmail.com]  
🔗 GitHub: [github.com/ictar](https://github.com/ictar)

## My Experience

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
            <div class="legend-title">📍 My Experience</div>
            <div class="legend-item">
                <i class="fa fa-graduation-cap" style="color:#4A90E2;margin-right:6px;"></i> Study
            </div>
            <div class="legend-item">
                <i class="fa fa-briefcase" style="color:#D0021B;margin-right:6px;"></i> Work
            </div>
            <div class="legend-item">
                <i class="fa fa-route" style="color:#7ED321;margin-right:6px;"></i> Travel
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