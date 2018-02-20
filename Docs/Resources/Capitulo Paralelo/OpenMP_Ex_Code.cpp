





			int main(){
				#pragma omp parallel for
				for(int i = 0; i < N_ITERATIONS; i++)
					func(...);
				return 0;
			}