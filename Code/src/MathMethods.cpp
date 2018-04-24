#include "MathMethods.h"

// Return least common multiple of two integers
int lcm(int num1, int num2){
    // Find the greater value of the two
    int max = (num1 > num2) ? num1 : num2;

    do{
        if(max % num1 == 0 && max % num2 == 0)
            return max;

        max++;
    } while(max < num1 * num2);

    // Insuccess
    return num1 * num2;
}
