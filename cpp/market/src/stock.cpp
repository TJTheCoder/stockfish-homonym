#include "stock.h"

using namespace std;

Stock::Stock (const string &symbol, const string &name, float initial_price):
    symbol(symbol),
    name(name),
    price(initial_price)
    {
        price_history.push_back(initial_price);
    }

void Stock::brownian_motion () {
    stock_mutex.lock();
    float change = (rand() % 100 - 50) / 100.0f;
    // price cannot drop to negative values
    while (price + change < 0) {
        change = (rand() % 100 - 50) / 100.0f;
    }
    price = price + change;
    price_history.push_back(price);
    stock_mutex.unlock();
}

float Stock::standard_deviation () {
    stock_mutex.lock();
    // calculate the mean
    float mean = 0;
    for (float price : price_history) {
        mean += price;
    }
    mean /= price_history.size();
    float variance = 0;
    for (float price : price_history) {
        // add the squared deviations
        float difference = price - mean;
        variance += (difference * difference);
    }
    // divide by the size and square root to find the standard deviation
    variance /= price_history.size();
    variance = sqrt(variance);
    stock_mutex.unlock();
    return variance;
}


void Stock::print_stock () {
    cout << "Stock: " << name << " (" << symbol << ")\n";
    stock_mutex.lock();
    cout << "Current Price: $" << price << "\n";
    cout << "Price History: ";
    for (float prices : price_history) {
        cout << "$" << prices << ", ";
    }
    stock_mutex.unlock();
    cout << endl; 
}

string Stock::get_symbol () const {
    return symbol;
}

string Stock::get_name () const {
    return name;
}

float Stock::get_price () const {
    return price;
}

vector<float> Stock::get_price_history () const {
    return price_history;
}