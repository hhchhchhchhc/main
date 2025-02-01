import hashlib
import hmac
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from time import sleep
from urllib.parse import urlencode

import requests
import websocket

# 设置代理（如果需要）
proxy = "http://127.0.0.1:7890"
os.environ["https_proxy"] = proxy
os.environ["http_proxy"] = proxy


# ---------------------------
# 配置参数
# ---------------------------
class Config:
    API_KEY = "Gwcp8jbCNfEZXlYqJwSVwOyB5ZFRQCVOcAiciibTgxrd1QzJpcnbpNSoUrm7Bqqo"
    API_SECRET = "wocKkvTiHGszLtUqJ43YlDmmCSSt8ZOgMqQQVJ7kU2KULxiWFX9ZsfSXdiCalj1t"
    BASE_ASSET = "BTC"
    QUOTE_ASSET = "USDT"
    PRICE_FLOOR_PCT = Decimal("0.05")  # 买入价格下限浮动
    PRICE_CEILING_PCT = Decimal("0.05")  # 卖出价格上限浮动
    MAX_ORDER_AMOUNT = 300.0  # 单笔订单最大金额
    ORDER_REFRESH_SEC = 60  # 订单刷新时间
    SYMBOL = "BTCUSDT"  # 交易对


# ---------------------------
# 自定义异常
# ---------------------------
class BinanceRestException(Exception):
    def __init__(self, reason, status_code, http_method, path, details):
        self.reason = reason
        self.status_code = status_code
        self.http_method = http_method
        self.path = path
        self.details = details
        super().__init__(
            f"{http_method} {path} failed with status code {status_code}: {reason}. Details: {details}"
        )


# ---------------------------
# 交易所客户端
# ---------------------------
class EnhancedBinanceClient:
    endpoints = {
        "getExchangeInfo": {
            "http_method": "GET",
            "path": "/api/v3/exchangeInfo",
            "is_signed": False,
        },
        "getAccount": {
            "http_method": "GET",
            "path": "/api/v3/account",
            "is_signed": True,
        },
        "postOrder": {
            "http_method": "POST",
            "path": "/api/v3/order",
            "is_signed": True,
            "required_params": [
                "symbol",
                "side",
                "type",
                "timeInForce",
                "quantity",
                "price",
            ],
        },
        "deleteOpenOrders": {
            "http_method": "DELETE",
            "path": "/api/v3/openOrders",
            "is_signed": True,
            "required_params": ["symbol"],
        },
        "postUserDataStream": {
            "http_method": "POST",
            "path": "/api/v3/userDataStream",
            "is_signed": False,
        },
        "putUserDataStream": {
            "http_method": "PUT",
            "path": "/api/v3/userDataStream",
            "is_signed": False,
            "required_params": ["listenKey"],
        },
        "getOrderBook": {
            "http_method": "GET",
            "path": "/api/v3/depth",
            "is_signed": False,
            "required_params": ["symbol"],
        },
    }

    def __init__(self, base_url="https://testnet.binance.vision", key="", secret=""):
        self.base_url = base_url
        self.key = key
        self.secret = secret
        if not key or not secret:
            raise ValueError("API key and secret are required")
        if not isinstance(key, str) or not isinstance(secret, str):
            raise ValueError("API key and secret must be strings")

        self.symbol_info_cache = None
        self.last_exchange_info_update = None
        self.logger = logging.getLogger("APIClient")

    def get_symbol_info(self, symbol):
        if not self.symbol_info_cache or (
            datetime.now() - self.last_exchange_info_update
        ) > timedelta(minutes=5):
            self.logger.info("Updating exchange info...")
            response = self.request("getExchangeInfo")
            self.symbol_info_cache = {s["symbol"]: s for s in response["symbols"]}
            self.last_exchange_info_update = datetime.now()
        return self.symbol_info_cache.get(symbol)

    def _get_sign(self, data):
        return hmac.new(self.secret.encode(), data.encode(), hashlib.sha256).hexdigest()

    def _verify_endpoint(self, endpoint):
        if endpoint not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint} 不存在")

    def _verify_api_credentials(self, endpoint):
        if self.endpoints[endpoint]["is_signed"] and (not self.key or not self.secret):
            raise PermissionError("需要API key和secret")

    def _verify_parameters(self, endpoint, params):
        required = self.endpoints[endpoint].get("required_params", [])
        for p in required:
            if p not in params:
                raise ValueError(f"缺少必要参数: {p}")

    def _prepare_request(self, endpoint, params):
        self.logger.debug(f"准备 {endpoint} 请求")
        self._verify_endpoint(endpoint)
        self._verify_api_credentials(endpoint)
        self._verify_parameters(endpoint, params)

        prepared_params = params.copy()
        if self.endpoints[endpoint]["is_signed"]:
            prepared_params["timestamp"] = self.get_timestamp()
            query_string = urlencode(prepared_params)
            prepared_params["signature"] = self._get_sign(query_string)

        return prepared_params

    def request(self, endpoint, params=None):
        params = params or {}
        self.logger.info(f"处理 {endpoint} 请求")

        try:
            self._verify_endpoint(endpoint)
            self._verify_api_credentials(endpoint)
            self._verify_parameters(endpoint, params)

            prepared_params = self._prepare_request(endpoint, params)
            endpoint_config = self.endpoints[endpoint]
            url = f"{self.base_url}{endpoint_config['path']}"
            headers = {"X-MBX-APIKEY": self.key}

            if endpoint == "deleteOpenOrders":
                if "symbol" not in prepared_params:
                    prepared_params["symbol"] = Config.SYMBOL

            if endpoint_config["http_method"] in ["GET", "DELETE"]:
                response = requests.request(
                    endpoint_config["http_method"].lower(),
                    url,
                    headers=headers,
                    params=prepared_params,
                    timeout=10,
                )
            else:
                response = requests.request(
                    endpoint_config["http_method"].lower(),
                    url,
                    headers=headers,
                    data=prepared_params,
                    timeout=10,
                )

            if not response.ok:
                raise BinanceRestException(
                    reason=response.reason,
                    status_code=response.status_code,
                    http_method=endpoint_config["http_method"],
                    path=endpoint_config["path"],
                    details=response.json(),
                )

            self.logger.info(f"{endpoint} 请求成功")
            return response.json()

        except BinanceRestException as e:
            self.logger.error(f"请求失败: {e.reason}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"网络请求异常: {str(e)}")
            raise

    def get_timestamp(self):
        return int(time.time() * 1000)


# ---------------------------
# 核心做市逻辑
# ---------------------------
class EnhancedMarketMaker:
    def __init__(self, client, symbol):
        self.client = client
        self.symbol = symbol
        self.orderbook = {"bids": [], "asks": []}
        self.listen_key = None
        self.ws = None
        self.logger = logging.getLogger("MarketMaker")
        self.running = True
        self.last_order_update = datetime.now()

        self.symbol_info = self.client.get_symbol_info(symbol)
        self.base_asset = self.symbol_info["baseAsset"]
        self.quote_asset = self.symbol_info["quoteAsset"]
        self.logger.info(f"交易对: {symbol} ({self.base_asset}/{self.quote_asset})")

        self.lot_size_filter = next(
            f for f in self.symbol_info["filters"] if f["filterType"] == "LOT_SIZE"
        )
        self.price_filter = next(
            f for f in self.symbol_info["filters"] if f["filterType"] == "PRICE_FILTER"
        )

        self.step_size = Decimal(self.lot_size_filter["stepSize"])
        self.tick_size = Decimal(self.price_filter["tickSize"])
        self.min_qty = Decimal(self.lot_size_filter["minQty"])
        self.min_price = Decimal(self.price_filter["minPrice"])

        self.init_user_data_stream()
        self.ws_thread = threading.Thread(target=self.run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        self.initialize_orderbook()

    def init_user_data_stream(self):
        try:
            response = self.client.request("postUserDataStream")
            self.listen_key = response["listenKey"]
            self.logger.info(f"Listen key: {self.listen_key}")
            self.start_listenkey_refresh()
            self.connect_user_data_ws()
        except Exception as e:
            self.logger.error(f"初始化用户数据流失败: {str(e)}")

    def start_listenkey_refresh(self):
        def refresh_task():
            while self.running:
                try:
                    sleep(30 * 60)
                    self.logger.info("刷新listen key...")
                    self.client.request(
                        "putUserDataStream", {"listenKey": self.listen_key}
                    )
                except Exception as e:
                    self.logger.error(f"刷新失败: {str(e)}")

        threading.Thread(target=refresh_task, daemon=True).start()

    def connect_user_data_ws(self):
        ws_url = f"wss://testnet.binance.vision/ws/{self.listen_key}"
        self.logger.info(f"连接用户数据流: {ws_url}")

        def on_message(ws, message):
            data = json.loads(message)
            if data["e"] == "executionReport":
                self.handle_order_update(data)

        def on_error(ws, error):
            self.logger.error(f"Websocket错误: {str(error)}")

        def on_close(ws, close_status_code, close_msg):
            self.logger.warning(f"连接关闭 ({close_status_code}: {close_msg})")

        self.ws = websocket.WebSocketApp(
            ws_url, on_message=on_message, on_error=on_error, on_close=on_close
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def handle_order_update(self, data):
        status = data["X"]
        order_id = data["i"]
        symbol = data["s"]
        side = data["S"]
        price = Decimal(data["L"])
        quantity = Decimal(data["l"])

        self.logger.info(f"订单更新: {status} {side} {quantity}@{price} ({order_id})")
        if status in ["FILLED", "PARTIALLY_FILLED"]:
            self.last_order_update = datetime.now()

    def truncate(self, value: Decimal, precision: Decimal) -> Decimal:
        return value.quantize(precision, rounding=ROUND_DOWN)

    def get_balance(self, asset):
        try:
            account = self.client.request("getAccount")
            balance = next(
                (
                    Decimal(b["free"])
                    for b in account["balances"]
                    if b["asset"] == asset
                ),
                Decimal(0),
            )
            self.logger.debug(f"{asset} 余额: {balance}")
            return balance
        except Exception as e:
            self.logger.error(f"获取余额失败: {str(e)}")
            return Decimal(0)

    def cancel_all_orders(self):
        try:
            self.client.request("deleteOpenOrders", {"symbol": self.symbol})
            self.logger.info("已取消所有订单")
        except BinanceRestException as e:
            if e.details.get("code") == -2011:
                self.logger.info("没有需要取消的订单")
            else:
                self.logger.error(f"取消订单失败: {str(e)}")

    def calculate_mid_price(self):
        try:
            bids = self.orderbook["bids"]
            asks = self.orderbook["asks"]

            best_bid = bids[0][0] if bids else None
            best_ask = asks[0][0] if asks else None

            if not best_bid or not best_ask:
                self.logger.warning("缺少买卖盘价格")
                return None

            mid = (best_bid + best_ask) / 2
            self.logger.debug(f"中间价计算: {mid} (买一: {best_bid}, 卖一: {best_ask})")
            return mid
        except IndexError:
            self.logger.warning("订单簿数据异常，无法计算中间价")
            return None

    def generate_orders(self):
        mid_price = self.calculate_mid_price()
        if not mid_price:
            return []

        floor_pct = Config.PRICE_FLOOR_PCT
        ceiling_pct = Config.PRICE_CEILING_PCT

        buy_price = mid_price * (Decimal("1") - floor_pct)
        buy_price = round(buy_price / self.tick_size) * self.tick_size

        sell_price = mid_price * (Decimal("1") + ceiling_pct)
        sell_price = round(sell_price / self.tick_size) * self.tick_size

        if buy_price < self.min_price or sell_price < self.min_price:
            self.logger.warning("价格低于最小允许值")
            return []

        base_balance = self.get_balance(self.base_asset)
        quote_balance = self.get_balance(self.quote_asset)

        max_buy_qty = min(
            quote_balance / buy_price, Decimal(str(Config.MAX_ORDER_AMOUNT)) / buy_price
        )
        max_buy_qty = round(max_buy_qty / self.step_size) * self.step_size

        max_sell_qty = min(
            base_balance, Decimal(str(Config.MAX_ORDER_AMOUNT)) / sell_price
        )
        max_sell_qty = round(max_sell_qty / self.step_size) * self.step_size

        orders = []
        if max_buy_qty >= self.min_qty:
            orders.append(("BUY", max_buy_qty, buy_price))
        if max_sell_qty >= self.min_qty:
            orders.append(("SELL", max_sell_qty, sell_price))

        self.log_order_details(
            mid_price, buy_price, sell_price, max_buy_qty, max_sell_qty
        )
        return orders

    def log_order_details(self, mid, buy_price, sell_price, buy_qty, sell_qty):
        self.logger.info("\n" + "=" * 40)
        self.logger.info(f"中间价: {mid}")
        self.logger.info(f"买入价: {buy_price} ({Config.PRICE_FLOOR_PCT*100}% 下方)")
        self.logger.info(f"卖出价: {sell_price} ({Config.PRICE_CEILING_PCT*100}% 上方)")
        self.logger.info(f"买入量: {buy_qty} {self.base_asset}")
        self.logger.info(f"卖出量: {sell_qty} {self.base_asset}")

        # 获取当前余额
        quote_balance = self.get_balance(self.quote_asset)
        base_balance = self.get_balance(self.base_asset)

        # 计算总资产价值（以USDT计价）
        total_value = quote_balance + (base_balance * mid if mid else Decimal(0))

        self.logger.info(f"法币余额: {quote_balance} {self.quote_asset}")
        self.logger.info(f"币余额: {base_balance} {self.base_asset}")
        self.logger.info(
            f"总资产估值: {total_value.quantize(Decimal('0.01'))} {self.quote_asset}"
        )
        self.logger.info("=" * 40 + "\n")

    def place_orders(self, orders):
        placed_orders = []
        for side, qty, price in orders:
            try:
                formatted_qty = str(self.truncate(qty, self.step_size))
                formatted_price = str(self.truncate(price, self.tick_size))

                order = self.client.request(
                    "postOrder",
                    {
                        "symbol": self.symbol,
                        "side": side,
                        "type": "LIMIT",
                        "timeInForce": "GTC",
                        "quantity": formatted_qty,
                        "price": formatted_price,
                    },
                )
                placed_orders.append(order)
                self.logger.info(
                    f"成功下单 {side}: {order['orderId']} "
                    f"{order['origQty']} @ {order['price']}"
                )
            except Exception as e:
                self.handle_order_error(e, side, qty, price)
        return placed_orders

    def handle_order_error(self, error, side, qty, price):
        if hasattr(error, "details"):
            error_code = error.details.get("code")
            if error_code == -2010:
                self.logger.warning(f"余额不足 {side} 订单: {qty}@{price}")
        else:
            self.logger.error(f"订单错误: {str(error)}")

    def initialize_orderbook(self):
        try:
            params = {"symbol": self.symbol, "limit": 100}
            response = self.client.request("getOrderBook", params)
            self.orderbook["bids"] = sorted(
                [(Decimal(b[0]), Decimal(b[1])) for b in response["bids"]],
                key=lambda x: -x[0],
            )[:100]

            self.orderbook["asks"] = sorted(
                [(Decimal(a[0]), Decimal(a[1])) for a in response["asks"]],
                key=lambda x: x[0],
            )[:100]
            self.logger.info(
                f"订单簿初始化完成，买盘档数: {len(self.orderbook['bids'])} 卖盘档数: {len(self.orderbook['asks'])}"
            )
        except Exception as e:
            self.logger.error(f"初始化订单簿失败: {str(e)}")
            raise

    def update_orderbook(self, side, updates):
        book = self.orderbook[side]
        is_bid = side == "bids"

        for price_str, qty_str in updates:
            price = Decimal(price_str)
            qty = Decimal(qty_str)

            if qty == 0:
                book[:] = [entry for entry in book if entry[0] != price]
                continue

            found = False
            for i, (existing_price, _) in enumerate(book):
                if existing_price == price:
                    book[i] = (price, qty)
                    found = True
                    break
                if (is_bid and existing_price < price) or (
                    not is_bid and existing_price > price
                ):
                    break

            if not found:
                insert_pos = 0
                while insert_pos < len(book):
                    current_price = book[insert_pos][0]
                    if (is_bid and current_price < price) or (
                        not is_bid and current_price > price
                    ):
                        break
                    insert_pos += 1
                book.insert(insert_pos, (price, qty))

        if is_bid:
            book.sort(key=lambda x: -x[0])
            if len(book) > 100:
                book[:] = book[:100]
        else:
            book.sort(key=lambda x: x[0])
            if len(book) > 100:
                book[:] = book[:100]

        self.orderbook[side] = book

    def run_websocket(self):
        ws_url = f"wss://testnet.binance.vision/ws/{self.symbol.lower()}@depth@100ms"
        self.logger.info(f"连接市场数据流: {ws_url}")

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get("e") == "depthUpdate":
                    event_time = datetime.fromtimestamp(data["E"] / 1000)
                    self.logger.debug(f"收到深度更新事件 ({event_time})")

                    if "b" in data:
                        self.update_orderbook("bids", data["b"])
                    if "a" in data:
                        self.update_orderbook("asks", data["a"])

                    best_bid = (
                        self.orderbook["bids"][0][0] if self.orderbook["bids"] else None
                    )
                    best_ask = (
                        self.orderbook["asks"][0][0] if self.orderbook["asks"] else None
                    )
                    self.logger.debug(
                        f"更新后订单簿 - 买一: {best_bid} 卖一: {best_ask}"
                    )

            except Exception as e:
                self.logger.error(f"处理市场数据错误: {str(e)}", exc_info=True)

        def on_error(ws, error):
            self.logger.error(f"市场数据错误: {str(error)}")

        def on_close(ws, close_status_code, close_msg):
            self.logger.warning(f"市场数据流关闭 ({close_status_code}: {close_msg})")
            if self.running:
                sleep(1)
                self.run_websocket()

        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        while self.running:
            try:
                self.ws.run_forever()
            except Exception as e:
                self.logger.error(f"Websocket错误: {str(e)}")
                sleep(1)

    def run(self):
        self.logger.info("启动做市程序...")
        consecutive_errors = 0

        while self.running:
            try:
                if (
                    datetime.now() - self.last_order_update
                ).seconds >= Config.ORDER_REFRESH_SEC:
                    self.logger.info("刷新订单...")

                    self.cancel_all_orders()
                    orders = self.generate_orders()
                    if orders:
                        self.place_orders(orders)
                        self.last_order_update = datetime.now()
                        consecutive_errors = 0
                    else:
                        self.logger.warning("未生成有效订单")

                sleep(1)

            except KeyboardInterrupt:
                # 处理 KeyboardInterrupt，确保程序能够优雅关闭
                self.logger.info("正在关闭...")
                self.running = False
                self.ws.close()
                break

            except Exception as e:
                self.logger.error(f"主循环出现错误: {str(e)}")
                consecutive_errors += 1

                # 如果连续错误次数过多，暂停交易
                if consecutive_errors >= 5:
                    self.logger.critical("出现过多连续错误，暂停交易")
                    sleep(300)  # 暂停5分钟
                    consecutive_errors = 0
                else:
                    sleep(5)


# 主程序
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler("market_maker.log")],
    )
    logger = logging.getLogger("Main")

    # 根据是否是回测模式决定初始化方式
    BACKTEST_MODE = False  # 设置为 False 启用实盘交易，设置为 True 启用回测

    if BACKTEST_MODE:
        # 初始化回测引擎
        backtest_engine = BacktestEngine(
            start_date=datetime.now() - timedelta(days=30),
            initial_balance={"BTC": Decimal("1.0"), "USDT": Decimal("50000.0")},
        )
    else:
        backtest_engine = None

    # 初始化Binance客户端
    client = EnhancedBinanceClient(key=Config.API_KEY, secret=Config.API_SECRET)

    # 初始化市场做市程序
    maker = EnhancedMarketMaker(client, Config.SYMBOL)

    try:
        maker.run()  # 启动做市逻辑
    except KeyboardInterrupt:
        # 如果是回测模式，生成最终的表现图
        if BACKTEST_MODE:
            backtest_engine.plot_performance()

            # 获取回测结果
            metrics = backtest_engine.get_metrics()
            print("\n回测结果:")
            print(f"总交易次数: {len(backtest_engine.trades)}")
            print(
                f"总盈亏: {sum([trade['value'] for trade in backtest_engine.trades]):.2f}"
            )
            print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    except Exception as e:
        logger.error(f"发生致命错误: {str(e)}")
        sys.exit(1)
