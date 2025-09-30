"""Launch the TraderV4 FastAPI service and trading loop."""

from __future__ import annotations

import threading

import uvicorn

from Traderv4.api import build_trader, create_app


def run(host: str = "0.0.0.0", port: int = 8001, log_level: str = "info") -> None:
    trader = build_trader()
    app = create_app(trader, trader.state_store)

    stop_event = threading.Event()

    trader_thread = threading.Thread(
        target=trader.run_forever,
        kwargs={"stop_event": stop_event},
        name="trader-loop",
        daemon=False,
    )
    trader_thread.start()

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        server.force_exit = True
        stop_event.set()
        trader_thread.join(timeout=10)


if __name__ == "__main__":
    run()
