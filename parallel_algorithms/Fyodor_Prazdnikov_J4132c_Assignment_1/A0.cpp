#include <iostream>
#include <string>

using namespace std;

// Function to count the number of words in a string
int countWords(const string &inputString) {
    int wordCount = 0;
    bool inWord = false;
    for (char c : inputString) {
        if (c == ' ' || c == '\t' || c == '\n') {
            inWord = false;
        } else {
            if (!inWord) {
                wordCount++;
                inWord = true;
            }
        }
    }
    return wordCount;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <input_string>" << endl;
        return 1;
    }

    string inputString = argv[1];
    int wordCount = countWords(inputString);

    cout << "Number of words: " << wordCount << endl;
    return 0;
}
