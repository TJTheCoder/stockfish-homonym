#ifndef MARKET_H
#define MARKET_H

#include "utils.h"
#include "stock.h"
#include "orderbook.h"
#include "trader.h"

using namespace std;

enum class TraderAction {
    BUY_ORDER,
    SELL_ORDER,
    GET_STOCKS
};

class Market {
    private:
        vector<pair<Stock&, OrderBook>> market;
        unordered_map<int, Trader> traders;
        mutex market_mutex;

    public:
        void add_stock (Stock& stock, int market_cap);
        void add_trader (int socket_desc, float initial_balance);
        string add_order (int socket_desc, string request);

        string recommend_stocks (string request);
        void print_market ();

        // GETTERS
        vector<pair<Stock&, OrderBook>> get_market ();
        Trader& get_trader (int socket_desc);
        string get_trader_info (int socket_desc);
};

#endif // MARKET_H