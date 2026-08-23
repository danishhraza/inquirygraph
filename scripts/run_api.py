"""Run the InquiryGraph API server."""

import uvicorn

from inquirygraph.config.settings import settings


def main():
    uvicorn.run(
        "inquirygraph.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
