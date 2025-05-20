#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

/* Constants for plagiarism detection */
#define MAX_SENTENCES 1000       // Maximum number of sentences to process
#define MAX_LINE_LENGTH 1024     // Maximum length of a single line
#define SIMILARITY_THRESHOLD 0.7 // Threshold to consider content similar
#define PLAGIARISM_THRESHOLD 0.5 // Overall document similarity threshold for plagiarism

/**
 * Returns the minimum of two integers
 * @param a First integer
 * @param b Second integer
 * @return The smaller of a and b
 */
int min(int a, int b){
    if(a <= b)
        return a;
    return b;
}

/**
 * Calculates the edit distance (Levenshtein distance) between two strings
 * using dynamic programming approach
 * @param s1 First string
 * @param s2 Second string
 * @param m Length of first string
 * @param n Length of second string
 * @return Edit distance between s1 and s2
 */
int editDistance(char *s1, char *s2, int m, int n){
    // Create a DP table to store results of subproblems
    int dp[m+1][n+1];
    
    // Base cases: empty string transformations
    for(int i = 0; i <= m; i++)
        dp[i][0] = i;  // Cost of deleting i characters
    for(int i = 0; i <= n; i++)
        dp[0][i] = i;  // Cost of inserting i characters
    
    // Fill the DP table
    for(int i = 1; i <= m; i++){
        for(int j = 1; j <= n; j++){
            // If characters match, no operation needed
            if(s1[i-1] == s2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                // Take minimum of insert, delete, replace operations
                dp[i][j] = 1 + min(min(dp[i][j-1], dp[i-1][j]), dp[i-1][j-1]);
        }
    }
    return dp[m][n];  // Return the final edit distance
}

/**
 * Creates k-shingles (contiguous subsequences of k items) from an array of strings
 * @param str_arr Array of strings (sentences)
 * @param len_arr Array containing lengths of each string
 * @param shingleSize Size of each shingle (k)
 * @param lengthOfShingleList Output array to store number of shingles per sentence
 * @param numOfLines Number of sentences
 * @return 3D matrix of shingles for each sentence
 */
char *** makeShingles(char *str_arr[], int len_arr[], int shingleSize, int lengthOfShingleList[], int numOfLines){
    // Allocate memory for the shingles matrix
    char ***shinglesMatrix = (char ***)malloc(numOfLines * sizeof(char **));

    for(int z = 0; z < numOfLines; z++){
        // Calculate number of shingles for this sentence
        int row = len_arr[z] - shingleSize + 1;
        int column = shingleSize + 1;  // +1 for null terminator

        // Allocate memory for shingles of this sentence
        char** temp_shingle = (char**)malloc(row * sizeof(char*));
        for (int i = 0; i < row; i++) {
            temp_shingle[i] = (char*)malloc(column * sizeof(char));
        }

        // Create shingles by sliding window of size 'shingleSize'
        int count = 0;
        for(int i = 0; i < row; i++){
            int k = i;
            for(int j = i; j < i + shingleSize; j++){
                temp_shingle[i][j-i] = str_arr[z][k];
                k++;
            }
            temp_shingle[i][shingleSize] = '\0';  // Null-terminate the shingle
            count++;
        }
        
        lengthOfShingleList[z] = count;  // Store number of shingles for this sentence
        shinglesMatrix[z] = temp_shingle;
    }

    return shinglesMatrix;
}

/**
 * Creates sets of unique shingles by removing duplicates
 * @param setOfShingles 3D matrix of shingles
 * @param shingleSize Size of each shingle
 * @param lengthOfShingleList Array containing number of shingles per sentence
 * @param lengthOfShingleSet Output array to store number of unique shingles per sentence
 * @param numOfLines Number of sentences
 * @return 3D matrix of unique shingles for each sentence
 */
char *** makeSetForShingles(char ***setOfShingles, int shingleSize, int lengthOfShingleList[], int lengthOfShingleSet[], int numOfLines){
    // Simple hash table to track seen shingles
    int hash[1000] = {0}; 
    
    // Initialize hash table
    for(int i = 0; i <= 200; i++){
        hash[i] = 0;
    }

    // Process each sentence
    for(int i = 0; i < numOfLines; i++){
        int cnt = 0;
        // Process each shingle in the sentence
        for(int j = 0; j < lengthOfShingleList[i]; j++){
            char c[shingleSize];
            int val = 0;
            
            // Calculate a simple hash value for the shingle
            for(int k = 0; k < shingleSize; k++){
                val = val + (setOfShingles[i][j][k]) * (k+1);
                c[k] = setOfShingles[i][j][k];
            }
            
            // If this hash hasn't been seen before, mark it as unique
            if(hash[val] != 1){
                hash[val] = 1;
                cnt++;
            }
            else{
                // Mark duplicates for removal
                setOfShingles[i][j] = "-1";
            }
            val = 0;
        }
        lengthOfShingleSet[i] = cnt;  // Store number of unique shingles

        // Reset hash table for next sentence
        memset(hash, 0, sizeof(hash)); 
    }
    
    // Create new arrays with only unique shingles
    for (int i = 0; i < numOfLines; i++) {
        char** temp_shingle = (char**)malloc(lengthOfShingleSet[i] * sizeof(char*));
    
        for (int j = 0; j < lengthOfShingleSet[i]; j++) {
            temp_shingle[j] = (char*)malloc(shingleSize * sizeof(char));
        }

        int set_i = 0;
        // Copy only non-duplicate shingles
        for (int j = 0; j < lengthOfShingleList[i]; j++) {
            if (strcmp(setOfShingles[i][j], "-1") != 0) {                
                temp_shingle[set_i] = setOfShingles[i][j];
                set_i++;
            }
        }

        setOfShingles[i] = temp_shingle;
    }

    return setOfShingles;
}

/**
 * Creates a vocabulary set of all unique shingles across all sentences
 * @param listOfShingles 3D matrix of shingles
 * @param lengthOfShingleList Array containing number of shingles per sentence
 * @param shingleSize Size of each shingle
 * @param vocabSize Output parameter to store vocabulary size
 * @param numOfLines Number of sentences
 * @return Array of unique shingles (vocabulary)
 */
char** makeSetForVocabulary(char ***listOfShingles, int lengthOfShingleList[], int shingleSize, int *vocabSize, int numOfLines){
    // Allocate memory for vocabulary set (with generous size)
    char** vocabSet = (char**)malloc(10000 * sizeof(char*));

    // Initialize with the first shingle
    vocabSet[0] = (char*)malloc(shingleSize * sizeof(char));
    strcpy(vocabSet[0], listOfShingles[0][0]);
    
    int vocab_ind = 1;    

    // Process all shingles from all sentences
    for(int i = 0; i < numOfLines; i++){
        for(int j = 0; j < lengthOfShingleList[i]; j++){
            int flag = 0;
            // Check if shingle already exists in vocabulary
            for(int k = 0; k < vocab_ind; k++){
                if(strcmp(listOfShingles[i][j], vocabSet[k]) == 0){
                    flag = 1;
                    break;
                }
            }
            // If new shingle, add to vocabulary
            if(flag == 0){
                vocabSet[vocab_ind] = (char*)malloc(shingleSize * sizeof(char));
                strcpy(vocabSet[vocab_ind], listOfShingles[i][j]);
                vocab_ind++;
            }
        }
    }

    *vocabSize = vocab_ind;  // Update vocabulary size
    return vocabSet;
}

/**
 * Creates one-hot encoding vectors for each sentence based on vocabulary
 * @param vocabSet Vocabulary set of unique shingles
 * @param setOfShingles 3D matrix of shingles for each sentence
 * @param vocabSize Size of vocabulary
 * @param lengthOfShingleSet Array with number of unique shingles per sentence
 * @param numOfLines Number of sentences
 * @return 2D matrix of one-hot vectors for each sentence
 */
double **oneHotVectors(char **vocabSet, char ***setOfShingles, int vocabSize, int lengthOfShingleSet[], int numOfLines){
    // Allocate memory for vectors
    double **Vectors = (double**)malloc(numOfLines * sizeof(double*));
    for (int i = 0; i < numOfLines; i++)
        Vectors[i] = (double*)malloc(vocabSize * sizeof(double));

    // Initialize all vectors to 0
    for(int i = 0; i < numOfLines; i++){
        for(int j = 0; j < vocabSize; j++){
            Vectors[i][j] = 0;
        }
    }

    // Set 1 for each shingle present in the sentence
    for(int i = 0; i < numOfLines; i++){
        for(int j = 0; j < lengthOfShingleSet[i]; j++){
            for(int k = 0; k < vocabSize; k++){
                if(strcmp(vocabSet[k], setOfShingles[i][j]) == 0){
                    Vectors[i][k] = 1;
                }
            }
        }
    }

    return Vectors;    
}

/**
 * Shuffles an array randomly
 * @param array Array to shuffle
 * @param size Size of the array
 */
void shuffle(int array[], int size) {
    // Fisher-Yates shuffle algorithm
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

/**
 * Creates dense vectors using MinHash technique for Locality Sensitive Hashing (LSH)
 * @param sentenceVectors One-hot vectors for each sentence
 * @param vocabSize Size of vocabulary
 * @param numOfLines Number of sentences
 * @return 2D matrix of dense vectors for each sentence
 */
double **makeDenseVectors(double **sentenceVectors, int vocabSize, int numOfLines){
    // Matrix to store MinHash functions
    int minHashFun[numOfLines][vocabSize];

    // Allocate memory for dense vectors
    double **denseVectors = (double**)malloc(numOfLines * sizeof(double*));
    for (int i = 0; i < numOfLines; i++)
        denseVectors[i] = (double*)malloc(numOfLines * sizeof(double));

    // Initialize dense vectors to 0
    for(int i = 0; i < numOfLines; i++){
        for(int j = 0; j < numOfLines; j++){
            denseVectors[i][j] = 0.0;
        }
    }

    // Create array for permutation
    int arr[vocabSize];
    for (int i = 0; i < vocabSize; i++) {
        arr[i] = i + 1;
    }

    // Generate random permutations for MinHash functions
    for(int i = 0; i < numOfLines; i++){
        shuffle(arr, vocabSize);
        for(int j = 0; j < vocabSize; j++){
            minHashFun[i][j] = arr[j];
        }
    }

    // Apply MinHash to create dense vectors
    for(int i = 0; i < numOfLines; i++){
        for(int j = 0; j < numOfLines; j++) {
            int found = 0;  
            // Find the first 1 in the permuted order
            for(int search = 1; search <= vocabSize && !found; search++) {
                for(int k = 0; k < vocabSize; k++) {
                    if(minHashFun[j][k] == search) {
                        if(sentenceVectors[i][k] == 1) {
                            denseVectors[i][j] = k + 1;
                            found = 1; 
                            break; 
                        }
                    }
                }
            }
        }
    }

    return denseVectors;
}

/**
 * Calculates dot product of two vectors
 * @param A First vector
 * @param B Second vector
 * @param size Size of vectors
 * @return Dot product value
 */
double dot_product(double *A, double *B, int size) {
    double dot = 0.0;
    for (int i = 0; i < size; i++) {
        dot += A[i] * B[i];
    }
    return dot;
}

/**
 * Calculates the magnitude (Euclidean norm) of a vector
 * @param vector Input vector
 * @param size Size of vector
 * @return Magnitude value
 */
double magnitude(double *vector, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += vector[i] * vector[i];
    }
    return sqrt(sum);
}

/**
 * Calculates cosine similarity between two vectors
 * @param A First vector
 * @param B Second vector
 * @param size Size of vectors
 * @return Cosine similarity value (between 0 and 1)
 */
double cosine_similarity(double *A, double *B, int size) {
    double dot = dot_product(A, B, size);
    double mag_A = magnitude(A, size);
    double mag_B = magnitude(B, size);
    
    // Prevent division by zero
    if (mag_A == 0 || mag_B == 0)
        return 0;
        
    return dot / (mag_A * mag_B);
}

/**
 * Structure to represent a single sentence
 */
typedef struct {
    char *text;    // Text content
    int length;    // Length of the text
} Sentence;

/**
 * Structure to represent a document composed of sentences
 */
typedef struct {
    Sentence *sentences;   // Array of sentences
    int num_sentences;     // Number of sentences
    char *filename;        // Source filename
} Document; 

/**
 * Splits document text into individual sentences
 * @param doc Document structure to populate
 * @param text Raw text content
 */
void splitIntoSentences(Document *doc, char *text) {
    char *token;
    char delimiters[] = ".!?";  // Sentence-ending punctuation
    char *text_copy = strdup(text);
    char *saveptr;
    
    // Count the number of sentences
    int count = 0;
    for (int i = 0; text[i]; i++) {
        if (strchr(delimiters, text[i])) {
            count++;
        }
    }
    
    // Allocate space for sentences
    doc->sentences = (Sentence *)malloc(count * sizeof(Sentence));
    doc->num_sentences = 0;
    
    // Split text into sentences
    token = strtok_r(text_copy, delimiters, &saveptr);
    while (token != NULL && doc->num_sentences < count) {
        // Remove leading whitespace
        while (*token == ' ' || *token == '\n' || *token == '\r' || *token == '\t') {
            token++;
        }
        
        // Skip empty sentences
        if (strlen(token) > 0) {
            // Convert to lowercase for case-insensitive comparison
            for (int i = 0; token[i]; i++) {
                token[i] = tolower(token[i]);
            }
            
            doc->sentences[doc->num_sentences].text = strdup(token);
            doc->sentences[doc->num_sentences].length = strlen(token);
            doc->num_sentences++;
        }
        
        token = strtok_r(NULL, delimiters, &saveptr);
    }
    
    free(text_copy);
}

/**
 * Reads a file into memory
 * @param filename Path to the file
 * @return String containing file contents or NULL if error
 */
char *readFile(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    
    // Allocate memory for the entire file
    char *buffer = (char *)malloc(size + 1);
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    // Read file into buffer
    size_t result = fread(buffer, 1, size, file);
    if (result != size) {
        fprintf(stderr, "Reading error\n");
        free(buffer);
        fclose(file);
        return NULL;
    }
    
    buffer[size] = '\0';  // Null-terminate the string
    fclose(file);
    return buffer;
}

/**
 * Calculates similarity between two documents using LSH technique
 * @param doc1 First document
 * @param doc2 Second document
 * @return Similarity score between 0 and 1
 */
double documentSimilarityLSH(Document *doc1, Document *doc2) {
    int totalSentences = doc1->num_sentences + doc2->num_sentences;
    char *str_arr[totalSentences];
    int len_arr[totalSentences];
    
    // Combine sentences from both documents
    for (int i = 0; i < doc1->num_sentences; i++) {
        str_arr[i] = doc1->sentences[i].text;
        len_arr[i] = doc1->sentences[i].length;
    }
    
    for (int i = 0; i < doc2->num_sentences; i++) {
        str_arr[doc1->num_sentences + i] = doc2->sentences[i].text;
        len_arr[doc1->num_sentences + i] = doc2->sentences[i].length;
    }
    
    int shingleSize = 3;  // Size of k-shingles
    int lengthOfShingleList[totalSentences];
    int lengthOfShingleSet[totalSentences];
    int vocabSize = 1;
    
    // Process using LSH pipeline
    char ***listOfShingles = makeShingles(str_arr, len_arr, shingleSize, lengthOfShingleList, totalSentences);
    char **vocabSet = makeSetForVocabulary(listOfShingles, lengthOfShingleList, shingleSize, &vocabSize, totalSentences);
    char ***setOfShingles = makeSetForShingles(listOfShingles, shingleSize, lengthOfShingleList, lengthOfShingleSet, totalSentences);
    double **sentenceVectors = oneHotVectors(vocabSet, setOfShingles, vocabSize, lengthOfShingleSet, totalSentences);
    double **denseVectors = makeDenseVectors(sentenceVectors, vocabSize, totalSentences);
    
    // Calculate cross-document similarities
    int numSimilarSentencePairs = 0;
    int totalCrossPairs = doc1->num_sentences * doc2->num_sentences;
    
    // Compare each sentence from doc1 with each sentence from doc2
    for (int i = 0; i < doc1->num_sentences; i++) {
        for (int j = 0; j < doc2->num_sentences; j++) {
            int doc2_index = doc1->num_sentences + j;
            double similarity = cosine_similarity(denseVectors[i], denseVectors[doc2_index], totalSentences);
            
            // Count pairs that exceed similarity threshold
            if (similarity >= SIMILARITY_THRESHOLD) {
                numSimilarSentencePairs++;
            }
        }
    }
    
    // Calculate document similarity as proportion of similar sentence pairs
    double docSimilarity = 0.0;
    if (totalCrossPairs > 0) {
        docSimilarity = (double)numSimilarSentencePairs / totalCrossPairs;
    }
    
    // Free memory (note: this is not comprehensive)
    for (int i = 0; i < totalSentences; i++) {
        free(denseVectors[i]);
        free(sentenceVectors[i]);
    }
    free(denseVectors);
    free(sentenceVectors);
    
    return docSimilarity;
}

/**
 * Calculates similarity between two documents using Edit Distance
 * @param doc1 First document
 * @param doc2 Second document
 * @return Similarity score between 0 and 1
 */
double documentSimilarityEditDistance(Document *doc1, Document *doc2) {
    int totalSimilarityScore = 0;
    int totalComparisons = 0;
    
    // Compare each sentence from doc1 with each sentence from doc2
    for (int i = 0; i < doc1->num_sentences; i++) {
        for (int j = 0; j < doc2->num_sentences; j++) {
            // Calculate edit distance between sentences
            int editDist = editDistance(doc1->sentences[i].text, doc2->sentences[j].text, 
                                        doc1->sentences[i].length, doc2->sentences[j].length);
            
            // Calculate a normalized score (1 - normalized edit distance)
            int maxLength = (doc1->sentences[i].length > doc2->sentences[j].length) ? 
                            doc1->sentences[i].length : doc2->sentences[j].length;
            
            double normalizedEditDistance = (maxLength > 0) ? (double)editDist / maxLength : 0;
            double similarityScore = 1.0 - normalizedEditDistance;
            
            // Count pairs that exceed similarity threshold
            totalSimilarityScore += (similarityScore >= SIMILARITY_THRESHOLD) ? 1 : 0;
            totalComparisons++;
        }
    }
    
    // Return proportion of similar sentence pairs
    return (totalComparisons > 0) ? (double)totalSimilarityScore / totalComparisons : 0;
}

/**
 * Checks for plagiarism between two documents and writes report
 * @param doc1 First document
 * @param doc2 Second document
 * @param outputFile File to write report to
 */
void checkPlagiarism(Document *doc1, Document *doc2, FILE *outputFile) {
    // Calculate similarity using two different methods
    double lshSimilarity = documentSimilarityLSH(doc1, doc2);
    double editDistSimilarity = documentSimilarityEditDistance(doc1, doc2);
    
    // Average of both similarity measures
    double overallSimilarity = (lshSimilarity + editDistSimilarity) / 2.0;
    
    // Write report to file
    fprintf(outputFile, "----------------------------------------\n");
    fprintf(outputFile, "Plagiarism Detection Report\n");
    fprintf(outputFile, "----------------------------------------\n");
    fprintf(outputFile, "Document 1: %s\n", doc1->filename);
    fprintf(outputFile, "Document 2: %s\n", doc2->filename);
    fprintf(outputFile, "----------------------------------------\n");
    fprintf(outputFile, "LSH Similarity: %.4f\n", lshSimilarity);
    fprintf(outputFile, "Edit Distance Similarity: %.4f\n", editDistSimilarity);
    fprintf(outputFile, "Overall Similarity: %.4f\n", overallSimilarity);
    fprintf(outputFile, "----------------------------------------\n");
    
    // Determine if plagiarism is detected
    if (overallSimilarity >= PLAGIARISM_THRESHOLD) {
        fprintf(outputFile, "PLAGIARISM DETECTED: These documents show significant similarity.\n");
    } else {
        fprintf(outputFile, "NO PLAGIARISM DETECTED: These documents do not show significant similarity.\n");
    }
    fprintf(outputFile, "----------------------------------------\n\n");
}

/**
 * Processes a pair of files for plagiarism detection
 * @param file1 Path to first file
 * @param file2 Path to second file
 * @param outputFile File to write report to
 */
void processPair(const char *file1, const char *file2, FILE *outputFile) {
    // Read files
    char *content1 = readFile(file1);
    char *content2 = readFile(file2);
    
    if (!content1 || !content2) {
        fprintf(stderr, "Error reading files\n");
        if (content1) free(content1);
        if (content2) free(content2);
        return;
    }
    
    // Process documents
    Document doc1, doc2;
    doc1.filename = strdup(file1);
    doc2.filename = strdup(file2);
    
    // Split documents into sentences
    splitIntoSentences(&doc1, content1);
    splitIntoSentences(&doc2, content2);
    
    // Check for plagiarism
    checkPlagiarism(&doc1, &doc2, outputFile);
    
    // Cleanup
    free(content1);
    free(content2);
    free(doc1.filename);
    free(doc2.filename);
    
    for (int i = 0; i < doc1.num_sentences; i++) {
        free(doc1.sentences[i].text);
    }
    free(doc1.sentences);
    
    for (int i = 0; i < doc2.num_sentences; i++) {
        free(doc2.sentences[i].text);
    }
    free(doc2.sentences);
}

/**
 * Processes all files in a directory for plagiarism detection
 * @param dirPath Path to directory
 * @param outputFile File to write report to
 */
void processDirectory(const char *dirPath, FILE *outputFile) {
    DIR *dir;
    struct dirent *entry1, *entry2;
    
    dir = opendir(dirPath);
    if (!dir) {
        fprintf(stderr, "Cannot open directory '%s'\n", dirPath);
        return;
    }
    
    // Get all files in directory
    const int MAX_FILES = 100;
    char *files[MAX_FILES];
    int fileCount = 0;
    
    while ((entry1 = readdir(dir)) != NULL && fileCount < MAX_FILES) {
        // Skip . and .. directories
        if (strcmp(entry1->d_name, ".") == 0 || strcmp(entry1->d_name, "..") == 0)
            continue;
            
        char filePath[512];
        snprintf(filePath, sizeof(filePath), "%s/%s", dirPath, entry1->d_name);
        
        struct stat path_stat;
        stat(filePath, &path_stat);
        
        if (S_ISREG(path_stat.st_mode)) { // Check if it's a regular file
            files[fileCount++] = strdup(filePath);
        }
    }
    
    closedir(dir);
    
    // Compare each pair of files
    for (int i = 0; i < fileCount; i++) {
        for (int j = i + 1; j < fileCount; j++) {
            processPair(files[i], files[j], outputFile);
        }
    }
    
    // Cleanup
    for (int i = 0; i < fileCount; i++) {
        free(files[i]);
    }
}

/**
 * Processes sentences using LSH and writes similarity scores to output file
 * @param str_arr Array of strings (sentences)
 * @param len_arr Array containing lengths of each string
 * @param numOfLines Number of sentences
 */
void LSH(char *str_arr[], int len_arr[], int numOfLines){
    int shingleSize = 3;
    int lengthOfShingleList[numOfLines];
    int lengthOfShingleSet[numOfLines];
    int vocabSize = 1;
    
    // Process using LSH pipeline
    char ***listOfShingles = makeShingles(str_arr, len_arr, shingleSize, lengthOfShingleList, numOfLines);
    char **vocabSet = makeSetForVocabulary(listOfShingles, lengthOfShingleList, shingleSize, &vocabSize, numOfLines);
    char ***setOfShingles = makeSetForShingles(listOfShingles, shingleSize, lengthOfShingleList, lengthOfShingleSet, numOfLines);
    double **sentenceVectors = oneHotVectors(vocabSet, setOfShingles, vocabSize, lengthOfShingleSet, numOfLines);
    double **denseVectors = makeDenseVectors(sentenceVectors, vocabSize, numOfLines);

    // Open output file for appending
    FILE *file_a = fopen("Output.txt", "a"); 
    if (file_a == NULL) { 
        printf("Could not open file"); 
        return; 
    }

    // Write similarity scores to file
    fprintf(file_a, "\n\n\nCosine Similarity between sentences are : \n\n");
    for(int i = 0; i < numOfLines; i++){
        for(int j = i+1; j < numOfLines; j++){
            fprintf(file_a, "Cosine Similarity between Sentence %d and %d is %f.\n", 
                   i+1, j+1, cosine_similarity(denseVectors[i], denseVectors[j], numOfLines));
        }
    }

    fclose(file_a);
}

/**
 * Displays usage instructions for the program
 */
void displayUsage() {
    printf("Plagiarism Detection Tool\n");
    printf("-------------------------\n");
    printf("Usage:\n");
    printf("  1. Compare two specific files:\n");
    printf("     ./program -f <file1> <file2>\n\n");
    printf("  2. Compare all files in a directory:\n");
    printf("     ./program -d <directory_path>\n\n");
    printf("  3. Use classic mode (from original code):\n");
    printf("     ./program\n\n");
}

/**
 * Main function - entry point of the program
 */
int main(int argc, char *argv[]){
    // Initialize random seed
    srand(time(NULL));
    
    // Open output file
    FILE *outputFile = fopen("PlagiarismReport.txt", "w");
    if (!outputFile) {
        fprintf(stderr, "Cannot create output file\n");
        return 1;
    }
    
    // If no arguments, run the original functionality (classic mode)
    if (argc == 1) {
        FILE* file_r = fopen("Input.txt", "r");
        char line[256];
        int numOfLines = 10;

        char* str_arr[numOfLines];
        int len_arr[numOfLines];

        int i = 0;
        if (file_r != NULL) {
            // Read sentences from input file
            while (fgets(line, sizeof(line), file_r)) {
                // Skip first 3 characters (line numbers)
                char* sentence = line+3;

                // Convert to lowercase
                for(int i = 0; sentence[i]; i++){
                    sentence[i] = tolower(sentence[i]);
                }

                size_t len = strlen(sentence);
                
                // Remove leading space if present
                if(sentence[0] == ' '){
                    for(int i = 0; i < len-1; i++)
                        sentence[i] = sentence[i+1];

                    len = strlen(sentence); 
                }

                // Remove trailing newline
                sentence[len-1] = '\0';
                len = strlen(sentence); 

                len_arr[i] = len;

                // Store sentence in array
                str_arr[i] = (char*)malloc((len + 1) * sizeof(char));
                strcpy(str_arr[i], sentence);    

                i++;        
            }
            fclose(file_r);
        }
        else {
            fprintf(stderr, "Unable to open file!\n");
            fclose(outputFile);
            return 1;
        }

        // Calculate Edit Distance between sentences
        FILE *file_w = fopen("Output.txt", "w"); 
        if (file_w == NULL) { 
            printf("Could not open file"); 
            fclose(outputFile);
            return 1; 
        }

        fprintf(file_w, "Edit Distance between Sentences : \n\n");

        for(int i = 0; i < numOfLines; i++){
            for(int j = i+1; j < numOfLines; j++){
                int result = editDistance(str_arr[i], str_arr[j], len_arr[i], len_arr[j]);
                fprintf(file_w, "Edit Distance between sentence %d and %d is %d.\n", i+1, j+1, result);
            }
        }

        fclose(file_w);

        // Calculate LSH similarities
        LSH(str_arr, len_arr, numOfLines);
        
        fprintf(outputFile, "Running in classic mode. See Output.txt for results.\n");
    }
    // New functionality - compare specific files or directory
    else if (argc >= 3) {
        if (strcmp(argv[1], "-f") == 0 && argc == 4) {
            // Compare two specific files
            processPair(argv[2], argv[3], outputFile);
            printf("Plagiarism report saved to PlagiarismReport.txt\n");
        }
        else if (strcmp(argv[1], "-d") == 0 && argc == 3) {
            // Process all files in directory
            processDirectory(argv[2], outputFile);
            printf("Plagiarism report saved to PlagiarismReport.txt\n");
        }
        else {
            displayUsage();
        }
    }
    else {
        displayUsage();
    }
    
    fclose(outputFile);
    return 0;
}
