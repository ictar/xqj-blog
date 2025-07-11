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
        const map = L.map('map').setView([45.4782, 9.2276], 4);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 12,
        }).addTo(map);

        // 当前语言判断
        const lang = document.documentElement.lang.includes('zh') ? 'zh-CN' : 'en';

        // 自定义图标（按类型区分）
        const iconMap = {
            study: new L.Icon.Default(),
            work: new L.Icon({
                iconUrl: "https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-red.png",
                shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            }),
            travel: new L.Icon({
                iconUrl: "https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-green.png",
                shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            })
        };

        // 加载对应语言的 JSON 文件
        fetch(`/data/places.${lang}.json`)
        .then(res => res.json())
        .then(data => {
            data.forEach(place => {
            const marker = L.marker(place.coords, { icon: iconMap[place.type] });
            const popupHtml = `
                <div style="min-width:180px">
                <strong>${place.title}</strong><br/>
                <span>${place.desc}</span><br/>
                <small>${place.years}</small>
                </div> `;
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
            <span class="legend-icon" style="background-color: #4A90E2;"></span> Study
            </div>
            <div class="legend-item">
            <span class="legend-icon" style="background-color: #D0021B;"></span> Work
            </div>
            <div class="legend-item">
            <span class="legend-icon" style="background-color: #7ED321;"></span> Travel
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