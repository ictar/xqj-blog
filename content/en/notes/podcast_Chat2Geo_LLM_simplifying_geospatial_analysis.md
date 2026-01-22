---
title: "[Podcast Notes] Chat2Geo and the Power of LLMs"
date: 2025-07-02
description: "Summary and thoughts on a podcast episode about 'Chat2Geo and LLM empowerment in geospatial analysis'"
tags: ["Chat2Geo", "LLM", "Geo", "Podcast Notes", "notes", "podcast", "notellm", "satellite-image-deep-learning"]
---

🎧 [Click here to access the audio notes for this podcast in NotebookLM](https://notebooklm.google.com/notebook/3396b39f-24fd-439e-8b27-90d28e866872/audio)


---


This podcast outlines the Chat2Geo application developed by Shahab Jozdani's company, Georina, and the philosophy behind it. Chat2Geo is a web-based application designed to simplify remote sensing-based geospatial analysis through an intuitive chatbot interface. It utilizes Large Language Models (LLMs) to democratize geospatial analysis, making it accessible to users without a geospatial background. The application was born out of the need for simpler interfaces and was inspired by popular LLM models like ChatGPT, aiming to provide a familiar and barrier-free user experience.

## Main Themes and Key Points
### Democratization and Simplification of Geospatial Analysis
- **Core Mission:** The primary goal of Georina and Chat2Geo is to "democratize and simplify complex geospatial tasks."
- **Addressing Pain Points:** While existing geospatial tools (like QGIS) are powerful, they lack easy-to-use web applications, especially for non-expert users. Chat2Geo fills this gap, enabling "those without a geospatial background" to conduct analyses.

> "We found that people don't like to engage with more complicated interfaces, they want something simpler."

> "In the field of geosciences, geospatial data science, one of the most important things missing was a software that is first web-native, and then has all the main components of the final product."

### Intuitive Interface Inspired by LLMs

- **Familiarity is Key:** Chat2Geo's interface design aims to mimic popular chatbots like ChatGPT or Claude. "The experience is very, very familiar, just like when you open ChatGPT or Claude."
- **Seamless User Onboarding:** The goal is to minimize the time from "wanting to start using an application to actually using it." "People don't need any training because it's just a simple box and a chatbot, and then they can start using the application."
- **Single Prompt Operation:** Users can perform analytical tasks through a "single prompt" and ask follow-up questions as the system retains context.

> "One of the most positive feedbacks was how easy it is to use this system, how fast it is, and how intuitive the system is."

### Data Connection and Integration
Three major data sources:
- **Satellite Data:** Primarily from Google Earth Engine.
- **User Uploaded Data:** Includes user-defined areas of interest (like GeoJSON or Shapefile), as well as non-spatial text data (like reports, policies, etc.).
- **External Databases:** The ability to connect to Esri feature services and upcoming PostgreSQL connections.

### Knowledge Base (RAG System)
Chat2Geo employs a system based on **Retrieval-Augmented Generation (RAG)** to process and query large non-spatial documents uploaded by users.

- **How RAG Works:** Documents are encoded by the LLM into binary digits (vectors) and stored in a vector database. User queries are also encoded to retrieve relevant information from the vector database, which is then decoded to generate a response.
- **Scalability:** The RAG system "can easily scale."
- **Challenges:** How data is "chunked" is crucial, as improper chunking can lead to information loss, hallucinations, or incorrect responses. "It depends heavily on the data you have, the context, the documents you have."

> "We have three main sources of data... satellite data... data uploaded by the user to the system... text data, non-spatial data like reports, policies, and all non-spatial things."

> "We call it a knowledge base, which is a RAG-based system that can easily query large documents for the user."

### Navigating the Rapidly Evolving Tech Landscape
- **MCP Protocol:** Chat2Geo plans to adopt MCP (Model Context Protocol), a "very powerful protocol" that can automatically update APIs, reducing the workload for developers to maintain integrations. "You don't need to worry about that anymore because those APIs will update automatically."
- **Challenge of Continuous Updates:** For a small team like Georina, keeping up with AI technology and new advancements is "very challenging."
- **Selective Adoption:** The team addresses this challenge by "cherry-picking those features that are really powerful and useful" rather than blindly introducing everything new.

> "MCP, I think this is one of the very hot topics recently."
> "For a small team like ours, keeping up with new technologies, new advancements is very, very challenging."

### LLM Applications in Software Development (e.g., Cursor)
- **Improving Development Efficiency:** AI IDEs like Cursor can dive deep into the codebase, write unit tests, add features, and debug code, thereby "increasing your efficiency or productivity."
- **Importance of Prompt Engineering:** Despite their power, effectively utilizing these tools requires expertise in "prompt engineering." Without sufficient context or "system prompts," LLMs might "break your architecture," "create irrelevant folders," leading to a messy and error-prone codebase.

> "It can write tests... can add new features... can debug your code."
> "If you don't give it enough information, enough context... it will break your architecture, it will create irrelevant folders."

### Future Vision and Education
- **Open Source Version:** Releasing an open-source version of Chat2Geo is intended to help "people who need to build something similar, they can start from somewhere."
- **Educational Resources:** Jazdani plans to release tutorial videos sharing the application development process and demonstrating how to effectively use AI systems (like Cursor) to improve efficiency. "I hope to release some videos, tutorials about how we started this application, how we developed it, and put different pieces together."

> "I think since we have this open-source version of the app... I hope to release some videos, tutorials."

## Key Facts
- **Company/Founder:** Georina, founded by Shahab Jozdani.
- **Founder Background:** Master's and Bachelor's in Geoinformatics, PhD in Physical Geography, focused on remote sensing and AI.
- **Product:** Chat2Geo, a web-based geospatial analysis application featuring a chatbot interface.
- **Development History:** Evolved from an early AI platform Aras, redesigned due to user demand for simpler interfaces, and inspired by LLM models like ChatGPT.
- **Main Data Sources:** Google Earth Engine (satellite data), user-uploaded data (GeoJSON, Shapefile, text files), external databases (Esri feature services, planned support for PostgreSQL).
- **Core Technologies:** Large Language Models (LLM), Retrieval-Augmented Generation (RAG) system, Vector Database.
- **Planned Technology Adoption:** MCP Protocol.
- **Development Tools Mentioned:** Cursor (AI IDE).
- **User Feedback:** Primarily praised for its ease of use, speed, and intuitive interface.

## Conclusion
Chat2Geo represents a significant advancement in the field of geospatial analysis, leveraging the power of LLMs to make complex tasks accessible and understandable. By focusing on user experience, integrating diverse data sources, and adapting to rapidly evolving AI technologies, Georina is actively shaping the future of geospatial data science, making it more democratic and efficient. However, effective implementation still requires a certain understanding of system functionality and prompt engineering.

## Podcast Information
Date: July 2, 2025

<div class="substack-post-embed"><p lang="en">Chat2Geo and the Power of LLMs by Robin Cole</p><p>with Shahab Jozdani</p><a data-post-link href="https://www.satellite-image-deep-learning.com/p/chat2geo-and-the-power-of-llms">Read on Substack</a></div><script async src="https://substack.com/embedjs/embed.js" charset="utf-8"></script>
