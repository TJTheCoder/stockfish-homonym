#include "market.h"

using namespace std;

#define PORT 8000
#define CONNECTION_LIMIT 100

mutex print_mutex;

struct ClientArgs {
    int client_socket;
    Market *market;
};

void signal_handler(int signal) {
    if (signal == SIGINT) {
        cout << "\nShutting down the market simulation...\n";
        exit(0);
    }
}

// continuously updates stock prices until simulation ends
void *stock_function (void* args) {
    Stock *stock_ptr = static_cast<Stock*>(args);
    while (true) {
        stock_ptr->brownian_motion();
        // sleep for 1-15 seconds before updating the stock price
        sleep(rand() % 15 + 1);
    }
    return nullptr;
}

void *market_function (void* args) {
    Market *market = static_cast<Market*>(args);
    while (true) {
        print_mutex.lock();
        market->print_market();
        print_mutex.unlock();
        // sleep for 10 seconds before printing the market again
        sleep(10);
    }
    return nullptr;
}

void *client_function (void* args) {
    ClientArgs *client_args = static_cast<ClientArgs*>(args);
    int socket_desc = client_args->client_socket;
    Market *market = client_args->market;
    delete client_args;

    market->add_trader(socket_desc, 10000.0);
    while (true) {
        char buffer[1024] = {0};
        int bytes_read = read(socket_desc, buffer, 1024);
        if (bytes_read <= 0) {
            print_mutex.lock();
            cout << "Client disconnected: " << socket_desc << "\n";
            print_mutex.unlock();
            close(socket_desc);
            break;
        }

        string request(buffer);
        string response;
        if (request == "GET_STOCKS") {
            response = market->get_trader_info(socket_desc);
        } else if (request.rfind("RECOMMEND", 0) == 0) {
            response = market->recommend_stocks(request);
        } else {
            response = market->add_order(socket_desc, request);
        }
        send(socket_desc, response.c_str(), response.size(), 0);
    }

    return nullptr;
}

int main () {

    // set up signal to end simulation
    signal(SIGINT, signal_handler);

    // define the stocks within the market
    Market market;

    Stock apple("AAPL", "Apple Inc.", 259.28);
    Stock tesla("TSLA", "Tesla Inc.", 454.13);
    Stock amazon("AMZN", "Amazon.com Inc.", 227.62);
    Stock google("GOOG", "Alphabet Inc. Class C", 197.48);
    Stock microsoft("MSFT", "Microsoft Corp.", 438.94);
    Stock nvidia("NVDA", "NVIDIA Corp.", 137.09);
    Stock gamestop("GME", "GameStop Corp.", 32.20);
    Stock intel("INTC", "Intel Corp.", 20.32);
    Stock disney("DIS", "Walt Disney Co.", 111.55);
    Stock paypal("PYPL", "PayPal Holdings Inc.", 86.86);

    market.add_stock(apple, 1000);
    market.add_stock(tesla, 1000);
    market.add_stock(amazon, 1000);
    market.add_stock(google, 1000);
    market.add_stock(microsoft, 1000);
    market.add_stock(nvidia, 1000);
    market.add_stock(gamestop, 1000);
    market.add_stock(intel, 1000);
    market.add_stock(disney, 1000);
    market.add_stock(paypal, 1000);

    // create realtime fluctuations in stock prices using multithreading
    pthread_t apple_thread, tesla_thread, amazon_thread, google_thread, microsoft_thread,
              nvidia_thread, gamestop_thread, intel_thread, disney_thread, paypal_thread;
    pthread_create(&apple_thread, nullptr, stock_function, &apple);
    pthread_create(&tesla_thread, nullptr, stock_function, &tesla);
    pthread_create(&amazon_thread, nullptr, stock_function, &amazon);
    pthread_create(&google_thread, nullptr, stock_function, &google);
    pthread_create(&microsoft_thread, nullptr, stock_function, &microsoft);
    pthread_create(&nvidia_thread, nullptr, stock_function, &nvidia);
    pthread_create(&gamestop_thread, nullptr, stock_function, &gamestop);
    pthread_create(&intel_thread, nullptr, stock_function, &intel);
    pthread_create(&disney_thread, nullptr, stock_function, &disney);
    pthread_create(&paypal_thread, nullptr, stock_function, &paypal);

    // create a market thread to print the market regularly
    pthread_t market_thread;
    pthread_create(&market_thread, nullptr, market_function, &market);

    // set up the socket connection
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // create server socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    // bind socket to a port
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*) &address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // listen for incoming connections
    if (listen(server_fd, CONNECTION_LIMIT) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    print_mutex.lock();
    cout << "Server listening on port " << PORT << "\n\n";
    print_mutex.unlock();

    // accept new connections 
    vector<pthread_t> threads;
    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr*) &address, (socklen_t*) &addrlen)) < 0) {
            perror("Accept failed");
            exit(EXIT_FAILURE);
        }

        print_mutex.lock();
        cout << "New connection: " << new_socket << "\n\n";
        print_mutex.unlock();

        pthread_t client_thread;
        ClientArgs *args = new ClientArgs{new_socket, &market};
        if (pthread_create(&client_thread, nullptr, client_function, args) != 0) {
            perror("Thread creation failed");
            delete args;
            continue;
        }

        threads.push_back(client_thread);
    }

    // close any current connections, the server, and stock threads
    for (pthread_t thread : threads) {
        pthread_join(thread, nullptr);
    }

    close(server_fd);

    pthread_join(apple_thread, nullptr);
    pthread_join(tesla_thread, nullptr);
    pthread_join(amazon_thread, nullptr);
    pthread_join(google_thread, nullptr);
    pthread_join(microsoft_thread, nullptr);
    pthread_join(nvidia_thread, nullptr);
    pthread_join(gamestop_thread, nullptr);
    pthread_join(intel_thread, nullptr);
    pthread_join(disney_thread, nullptr);
    pthread_join(paypal_thread, nullptr);

    return 0;
}