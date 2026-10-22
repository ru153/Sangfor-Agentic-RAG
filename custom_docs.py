from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from graph_build import build_agent
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="深信服训练营 Agentic RAG 接口",
    docs_url=None
)
agent = build_agent()

# 不使用 FastAPI 的封装，直接返回原生 HTML
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{app.title} - Swagger UI</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
        <link rel="icon" type="image/png" href="https://fastapi.tiangolo.com/img/favicon.png" />
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
        <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: "{app.openapi_url}",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                layout: "BaseLayout"
            }});
            window.ui = ui;
        }};
        </script>
    </body>
    </html>
    """
    # 加上防缓存头，强制浏览器每次都要获取最新页面
    return HTMLResponse(content=html_content, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})