// searcher1.cpp
#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include <string.h>

#include "PPbuf.h"
#include "dynstrlb.h"

void main(int argc, char **argv)
{
//char *filename;
//strcpy(filename,argv[1]); //*(argv+1));

char *searchstr;

   if (argc!=2)
   {
      cerr << "Usage: test1 file.xxx" << endl;
   //   exit(-1);
   }

   cout << "starting with text file: " << argv[1] << endl;
   PPbuf mybuf(argv[1]);   // Text file
   PPbuf empty, copy;
   String temp;
   String sentence;
  // char pause;


//   cout << "2nd char of the 3rd string is " << mybuf[2][1] << endl;




   // tokenize tests
//   sentence="Mary had a little lamb whose fleece was white as snow.";
  // PPbuf tokenbuf(sentence,".");
//   cout << "Sentence to tokenise is:" << endl;
//   cout << sentence << endl;
//   cout << "tokenbuf is:" << endl;
//   cout << tokenbuf;


   // search tests
 

 //  cout << mybuf;

cout << "What are you looking for?"; 
cin >> searchstr;

    PPbuf position;

//    cin >> pause;
//	mybuf.PPbufSearch(&position,"no");   // This function creates a search string specific
	// overlay in the buffer.   the results of this search can be accessed
	// by running the various overlay functions

  //  mybuf.PPbufSearch(&position,"is");
    mybuf.PPbufSearch(&position,searchstr);

	cout << endl; 
//	cout << "position=" << endl;
//	cout << position;
//	cout << endl;


	
	
	
	
	cout << "Finished search. Writing" << endl;

//cout << mybuf;
//cout << endl << endl << endl;

    mybuf.WriteTo("outfile.txt",0);  // new file

cout << "New done. now appending" << endl;

	position.WriteTo("outfile.txt",1);  // append file


	cout << "finished writing outfile.txt" << endl;

//	PPbuf tester1(position[0],",");
 //  cout << "Buf to tokenise is:" << endl;
//   cout << position[0] << endl;
//   cout << "tester1 is:" << endl;
//   cout << tester1;










}



