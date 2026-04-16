#include "orderbook.h"
#include "market.h"

using namespace std;

OrderBook::OrderBook (const string& stock_symbol):
    stock_symbol(stock_symbol)
    {}

void OrderBook::add_order (const Order& order, Market& market) {
    if (order.get_order_type() == OrderType::BUY) {
        buy_orders.push(order);
    } else {
        sell_orders.push(order);
    }
    match_orders(market);
}

void OrderBook::match_orders (Market& market) {
    while (buy_orders.size() > 0 && sell_orders.size() > 0) {
        const Order& top_buy = buy_orders.top();
        const Order& top_sell = sell_orders.top();

        // Match the order is greatest buy is larger than smallest sell
        if (top_buy.get_price() >= top_sell.get_price()) {
            int trade_quantity = min(top_buy.get_quantity(), top_sell.get_quantity());
            float trade_price = top_sell.get_price();

            // Update balances/stock quantities for the traders
            Trader& tb = market.get_trader(top_buy.get_trader_id());
            tb.update_balance(-trade_quantity * trade_price);
            tb.update_quantity(stock_symbol, trade_quantity);
            
            Trader& ts = market.get_trader(top_sell.get_trader_id());
            ts.update_balance(trade_quantity * trade_price);
            ts.update_quantity(stock_symbol, -trade_quantity);

            // Update the order books
            Order new_buy = top_buy;
            new_buy.reduce_quantity(trade_quantity);
            buy_orders.pop();
            // If order was partially filled, add it back to the queue
            if (new_buy.get_quantity() > 0) {
                buy_orders.push(new_buy);
            }

            Order new_sell = top_sell;
            new_sell.reduce_quantity(trade_quantity);
            sell_orders.pop();
            if (new_sell.get_quantity() > 0) {
                sell_orders.push(new_sell);
            }

        } else {
            break;
        }
    }
}

pair<double, double> OrderBook::get_best_bid_ask () const {
    double best_bid = buy_orders.empty() ? 0.0 : buy_orders.top().get_price();
    double best_ask = sell_orders.empty() ? 0.0 : sell_orders.top().get_price();
    return {best_bid, best_ask};
}


void OrderBook::print_order_book () const {
    cout << "Order Book for " << stock_symbol << ":\n";

    // Print Buy Orders
    cout << "Buy Orders (Descending):\n";
    auto buy_copy = buy_orders;
    while (!buy_copy.empty()) {
        const Order& order = buy_copy.top();
        cout << "  Trader " << order.get_trader_id()
                  << " | Qty: " << order.get_quantity()
                  << " | Price: $" << order.get_price() << "\n";
        buy_copy.pop();
    }

    // Print Sell Orders
    cout << "Sell Orders (Ascending):\n";
    auto sell_copy = sell_orders;
    while (!sell_copy.empty()) {
        const Order& order = sell_copy.top();
        cout << "  Trader " << order.get_trader_id()
                  << " | Qty: " << order.get_quantity()
                  << " | Price: $" << order.get_price() << "\n";
        sell_copy.pop();
    }
}