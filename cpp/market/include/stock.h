#ifndef STOCK_H
#define STOCK_H

#include "utils.h"

using namespace std;

class Stock {
    private:
        string symbol;
        string name;
        float price;
        vector<float> price_history;
        mutex stock_mutex;
    
    public:
        Stock (const string &symbol, const string &name, float initial_price);

        void brownian_motion ();
        float standard_deviation ();
        void print_stock ();

        // GETTERS
        string get_symbol () const;
        string get_name () const;
        float get_price () const;
        vector<float> get_price_history () const;
};

#endif // STOCK_H