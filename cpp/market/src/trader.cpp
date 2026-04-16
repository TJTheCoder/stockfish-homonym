#include "trader.h"

Trader::Trader () :
    Trader(0.0)
    {}

Trader::Trader (float initial_balance):
    balance(initial_balance)
    {}

void Trader::update_balance (float amount) {
    balance += amount;
}

void Trader::update_quantity (string stock_symbol, int quantity) {
    if (stocks.find(stock_symbol) != stocks.end()) {
        stocks[stock_symbol] += quantity;
        if (stocks[stock_symbol] == 0) {
            stocks.erase(stock_symbol);
        }
    } else {
        add_stock (stock_symbol, quantity);
    }
}

void Trader::add_stock (string stock_symbol, int initial_quantity) {
    stocks[stock_symbol] = initial_quantity;
}

float Trader::get_balance () {
    return balance;
}

int Trader::get_quantity (string stock_symbol) {
    if (stocks.find(stock_symbol) != stocks.end()) {
        return stocks[stock_symbol];
    }
    return 0;
}

string Trader::get_stocks () {
    ostringstream oss;
    oss << "Balance: " << fixed << setprecision(2) << balance << "\n";
    oss << "Your stocks:";
    int count = 0;
    for (const auto& [stock_name, quantity] : stocks) {
        count++;
        oss << "\n" << quantity << " shares of " << stock_name;
    }
    if (count == 0) {
        oss << " N/A";
    }
    oss << "\n";
    return oss.str();
}
