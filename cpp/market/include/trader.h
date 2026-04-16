#ifndef TRADER_H
#define TRADER_H

#include "utils.h"

using namespace std;

class Trader {
    private:
        float balance;
        unordered_map<string, int> stocks;

    public:
        Trader ();
        Trader (float initial_balance);

        void update_balance (float amount);
        void update_quantity (string stock_symbol, int quantity);
        void add_stock (string stock_symbol, int initial_quantity);

        // GETTERS
        float get_balance ();
        int get_quantity (string stock_symbol);
        string get_stocks ();
};

#endif