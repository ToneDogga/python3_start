// PPbuf.cpp

#include <iostream.h>
#include <fstream.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <new.h>

#include "PPbuf.h"

//error PPbuf::PPerror;  // static error object

// no argument constructor
PPbuf::PPbuf() 
{
//	char pause;
	_len = 0;
	terminator = '\n';
	_Space = CHUNKSZ;
	_Strings = new String[CHUNKSZ];
	if (!_Strings)
	{
		//PPerror.StandardError(NOMEM,__FILE__,__LINE__);
		cout << "error";
		delete[] _Strings;
		exit(0);
	}

//	_overlay = new (char**[CHUNKSZ]);
//	if (!_overlay)
//	{
		//PPerror.StandardError(NOMEM,__FILE__,__LINE__);
//		cout << "error";
//		exit(0);
//	}
//
//    printf("before _overlay points to %x\n",_overlay);
//	printf("before _Strings points to %x\n",_Strings);

//	_overlay=(char***)&_Strings;  //The first overlay pointer now points
	// to the start of the first line
	// this now means that we can have a string of chars pointing to
	// each char in the buffer.

//    printf("After _overlay points to %x\n",_overlay);
//	printf("After _Strings points to %x\n",_Strings);
//
// cin >> pause;

}


// when we need to expand the size of the buffer
void PPbuf::GrowBuf()
{
//	cout << "Grow buf starting" << endl;
	String *tmp=0;  //BUFEND;
//	char ***Stmp=0;
	long i=0;
//	int j;

	tmp = new String[_Space + CHUNKSZ];
	if (!tmp)
	{
	//	PPerror.StandardError(NOMEM,__FILE__,__LINE__);
		cout << "error";
		delete[] tmp;
		exit(0);
	}
		
//	Stmp = new (char**[_Space + CHUNKSZ]);
//	if (!Stmp)
//	{
	//	PPerror.StandardError(NOMEM,__FILE__,__LINE__);
//		cout << "error";
//		exit(0);
//	}
		

	// you cannot do the memcpy because
	// I must call delete []
	for (i=0; i < _Space; i++)
	{
		tmp[i] = _Strings[i];
//		Stmp[i]=_overlay[i];

		//for (j=0;j<MAXLINELENGTH;j++)
		  // Stmp[i][j] = _overlay[i][j];
	}
	
	// arrays are allocated differently. if you allocate an
	// array with new[] you MUST free it with delete [].

//	cout << "Starting to Free memory for grow buf" << endl;
	delete [] _Strings;
//	delete [] _overlay;
//	cout << "Free mem for grow buf successful" << endl;
	
	_Strings = tmp;
//	_overlay = Stmp;

//	delete [] tmp;
//	delete [] Stmp;

	_Space += CHUNKSZ;
}

// destructor
PPbuf::~PPbuf()
{
//int i,j;

 //   cout << "PPbuf - Destructor starting" << endl;
	
		
	delete []_Strings;

//	cout << "PPbuf - _strings destructed" << endl;
//    delete []_overlay;	
	// destructor called automatically for each object in
	// the array!

//	cout << "PPbuf - _overlay destructed" << endl;
//	cout << "PPbuf - Destructor finished" << endl;
}

// one argument constructor
// initialize PPbuf object from an "unprotected" ppbuf
// unprotected ppbuf should be NULL-terminated
PPbuf::PPbuf(char **ppbuf, long pplen)
{
	long i=0;
//	int j;

//	cerr << "Starting the one arg file load PPbuf constructor" << endl;
	assert(ppbuf!=NULL);
		
	// create a new PPbuf based on the length of ppbuf
	if (!pplen)
		for (i=0; ppbuf[i]; i++, pplen++) ;
	
	_Strings = new String[pplen + 1];
	if (!_Strings)
	{
	//	PPerror.StandardError(NOMEM,__FILE__,__LINE__);
		cout << "error";
		exit(0);
	}

//	_overlay = new (char**[pplen + 1]);
//	if (!_overlay)
//	{
	//	PPerror.StandardError(NOMEM,__FILE__,__LINE__);
//		cout << "error";
//		exit(0);
//	}


//	_overlay=(char***)&_Strings;  //The first overlay pointer now points
	// to the start of the first line
	// this now means that we can have a string of chars pointing to
	// each char in the buffer.

		
	if (ppbuf)
	{
		for (i=0; i < pplen; i++)
		{
			_Strings[i] = ppbuf[i];
//			_overlay[i]=NULL;

			//for (j=0;j<MAXLINELENGTH;j++)
			//   _overlay[i][j] = NULL;
		}
	}
//	strcpy(_Strings[pplen],NULL);  // NULL terminate
	_Strings[pplen]=BUFEND;  // NULL terminate
//	_overlay[pplen]=NULL;
	_len = pplen;
	_Space = pplen + 1;
	terminator = '\n';

}

// initialize PPbuf from a file.  replaced "file2ppbuf"
PPbuf::PPbuf(char *filename)
{
//	cout << "Starting the one arg file load PPbuf constructor" << endl;
	char buf[MAXLINELENGTH];
	ifstream infile(filename);
	long i,count;
//	int j;
  //  int overlay_size=sizeof(_overlay);

//	cout << "Size of _overlay=" << overlay_size << endl;


//	char pause;
	
	if (!infile)
	{
	//	PPerror.StandardError(NOTOPEN,__FILE__,__LINE__);
	   cout << "error";
	   exit(0);
	}
	

//	cin >> pause;
	// count the lines in the file
	for (count = 0; infile; count++)
	{
//		cout << "Getting line:" << count << endl;
	//	cin >> pause;
		infile.getline(buf,MAXLINELENGTH);
  //      cout << "count=" << count << " buf=" << buf << endl; 
	}
	
	
//	cout << "Buffer loaded: " << count << " lines" << endl;
//	cin >> pause;
	// now get space for PPbuf
	_len = count;
	count++;

//	cout << "Allocate string mem" << endl;
	_Strings = new String[count];
	if (!_Strings)
	{
//		PPerror.StandardError(NOMEM,__FILE__,__LINE__);		
		cout << "error";
	    exit(0);
	}

//	cout << "allocate overlay mem" << endl;
//	_overlay = new (char**[count]);	
//	if (!_overlay)
//	{
//		PPerror.StandardError(NOMEM,__FILE__,__LINE__);		
//		cout << "error";
//	    exit(0);
//	}

//	cout << "alloacetion done. closing file" << endl;
	// close and reopen file
	infile.close();
//	cout << "file closed" << endl;
	infile.open(filename);
	if (!infile)
	{
//		PPerror.StandardError(NOREWIND,__FILE__,__LINE__);
	    cout << "error opening file" << endl;
		infile.close();
		exit(0);
	}
	
	// Fill PPbuf
//	cout << "Ready to fill PPBuf" << endl;
	i=0;
//	j=0;
	while (infile)
	{
		infile.getline(buf,MAXLINELENGTH);
		_Strings[i] = buf;
//		cout << "_strings[" << i << "]=" << _Strings[i] << endl;

//		_overlay[i]=NULL;
//		cout << "_overlay[" << i << "]=NULL" << endl;
	//	cout << "Starting overlay initisation loop" << endl;
	//	for(j=0;j<32;j++)
	//	{
	//	    cout << "before _overlay[" << i << "][" << j << "]=" << *(*(_overlay+i)+j) << endl;
		//	printf("_overlay[%ld][%d]=%x\n",i,j,_overlay[i][j]);
	//		_overlay[i][j]=NULL;
		//	printf("%x\n",*(*(_overlay+j)+i));

	//	    cout << "after _overlay[" << i << "][" << j << "]=" << *(*(_overlay+i)+j) << endl;
      //     cin >> pause;
	//	}
	//	cout << "NULL assignment _overlay loop finished" << endl;
		i++;
	}
	
//	cout << "PPbuf filled. i=" << i << " _len=" << _len << " count=" << count << endl;
	//strcpy(_Strings[_len],NULL);   // NULL terminate
	_Strings[_len]=BUFEND;  //NULL terminate
//	_overlay[_len]=NULL;
//	cout << "null terminated." << endl;
	terminator = '\n';
	
  //  cout << "Finishing the one arg file load PPbuf constructor" << endl;
    infile.close();
}


// used as a cheat to get the casting right on the successful search address in the buffer
//char*** pointercast(char **searchpos)
//{
//return (&searchpos);
//}








void PPbuf::WriteTo(char* filename, int append_flag) 
{
	long i=0;
	
	if (!filename)
	{
	//	PPerror.StandardError(NOTOPEN,__FILE__,__LINE__);
	   cout << "error";
	   exit(0);
	}

//	if (append_flag)
		ofstream outfile(filename,ios::out|ios::app);  //append

//	if (append_flag)
//	{
//		outfile.open(filename,ios::out|ios::app);  // append
//	}
//	else
//	{
//		outfile.open(filename,ios::out);
//	}


    outfile.setmode(filebuf::text);
	
	//    outfile.open(OutFile,ios::out);
//	cout << "Outfile " << filename << " opened successfully. " << _len << "lines." << endl;
	while(i < _len-1)
	{
		outfile.write(_Strings[i],strlen(_Strings[i]));
		outfile.write("\n",1);
	    i++;
	}
	outfile.close();
	
 //   cout << filename << " Write completed " << i << " lines." << endl;

}

////////////////////////////////////////////////////////////////////////















// this searches for the search string and creates an overlay of addresses
// with the positions of the start of the search chars in them
// for each character on each string on the text buffer
void PPbuf::PPbufSearch(PPbuf *position, char *search)
{


	// This function searches the text buffer for a user defined string
	// it marks the address of each point where it finds this string on 
	// each line in the array _overlay which is part of the PPbuf class.
  	char *temp=0;
    long cnt=0;
	int pos=0;
	char *tempstr=0;
	int templen=0, searchlen=0, totalcutoff=0;
	char *foundpos=NULL;
//	int createflag=0;
	char *output=" ";

   	long number=1;  // the count of instances on the string found

   	if (!search)
	{

//		PPerror.StandardError(NULLARG,__FILE__,__LINE__);
		cout << "string search blank";
		exit(0);
	}

    searchlen=strlen(search);

    while(cnt<_len)
	{  // strcspn
       totalcutoff=0;
	   tempstr=_Strings[cnt];
//	   cout << "starting tempstr=" << tempstr << endl;
	   templen=strlen(tempstr);
	
	   
	   
	   
	   
	   while (tempstr)
	   {
//	      cout << "starting check loop on each line.  cnt=" << cnt << " tempstr=" << tempstr << endl;
		  pos = strstr(tempstr,search)-tempstr;
		  if (pos>=0)
		  {
// found
		//	  if (!createflag)
		//	  {
		//		  createflag=1;
		//	      PPbuf position;
		//	  }
              
//			  foundpos=(char*)(&_Strings[cnt][pos+totalcutoff]);


               strcpy(output,"");
	        //  cout << "Foundpos:'" << search << "' [" << cnt << "][" << pos+totalcutoff << "] (" << foundpos << ")" << endl;
		//	  sprintf(output,"%s,%ld,%d,%x",search,cnt,pos+totalcutoff,foundpos);
			  sprintf(output,"\"%s\",%ld,%ld,%d",search,number,cnt,pos+totalcutoff);


			  number++;
//			  cout << "output=" << output << endl;
       //         if (createflag)
// Write to position buffer the position of the search

			  position->AppendString(output);




  		//	   (_overlay)[cnt][pos+totalcutoff]=&(_Strings[cnt][pos+totalcutoff]);
  			//   (_overlay)[cnt][pos+totalcutoff]=&(_Strings[cnt][pos+totalcutoff]);
//			  (_overlay)[cnt]=(char**)foundpos;
		//	  printf("printf _overlay=%s or %x\n",_overlay[cnt],_overlay[cnt]);
 // Overlay is a pointer to a pointer to a pointer to a POINTER!
			   // it contains the address of the starting point of each successful search
			   totalcutoff=totalcutoff+pos+searchlen;
		       tempstr=tempstr+pos+searchlen;   // cut temporary string down to remainder after the found search string
		       templen=strlen(tempstr);
		 //      cout << "cut down tempstr=" << tempstr << " templen=" << templen << endl;		    
			   //   cout << "templen=" << templen << endl;
		  }
		  else
		  {
	//		  cout << "pos <=0" << endl;
			tempstr=NULL;
		  }
	   }

//	_len++;
	//   cout << "Incrementing cnt" << endl;
    	cnt++;
	
//	delete temp; 
	delete tempstr;
	}
//		cout << "Search function finished" << endl;
	
}


// this constructor replaces toks_from_str()
PPbuf::PPbuf(String line, char *delimiters)
{
  	char *temp=0, *tok=0;
    long cnt=0;
//	int j;
   	
   	if (!line || !delimiters)
	{
//		PPerror.StandardError(NULLARG,__FILE__,__LINE__);
		cout << "error";
		exit(0);
	}

	// initialize PPbuf
	_len = 0;
	terminator = '\n';
	_Space = CHUNKSZ;
	_Strings = new String[CHUNKSZ];
	if (!_Strings)
	{
//		PPerror.StandardError(NOMEM,__FILE__,__LINE__);
		cout << "error";
		delete[] _Strings;
		exit(0);
	}

//	_overlay = new (char**[CHUNKSZ]);
//	if (!_overlay)
//	{
//		PPerror.StandardError(NOMEM,__FILE__,__LINE__);
//		cout << "error";
//		exit(0);
//	}

  
    temp = line.Strdup();
    tok = strtok(temp,delimiters);
    if (tok) _Strings[cnt] = tok; _len++;
    cnt++;
   
    while ( (tok = strtok(NULL,delimiters)) != NULL)
    {
    	if (_len == _Space)
    		this->GrowBuf();

   	    _Strings[cnt] = tok; 
//		_overlay[cnt]=NULL;
	//	for (j=0;j<MAXLINELENGTH;j++)
	//	   _overlay[cnt][j] = NULL;
		_len++;
	    cnt++;
    } // while strtok does not return NULL 

	delete temp; 
}


ostream &operator << (ostream &stream, PPbuf &obj)
{
	long i=0;
//	int j=0;
//	char pause;

//	stream << "Start strings:" << obj._Strings << endl;
//	stream << "Start overlay:" << obj._overlay << endl;

	// dump the ppbuf to the stream with the terminator.
	for (i=0; i < obj._len; i++)
	{
		stream << obj._Strings[i];
//        j=0;	
//		while ((obj._overlay[i][j]))
//		{
//			if (obj._overlay[i])
//		        stream << "Pos=[" << i <<"][" << j << "](" << &obj._overlay[i][j] << ")";
//		        stream << "Pos=[" << i <<"][" << j << "](" << obj._overlay[i] << ")";
		 //  obj._overlay[i][j];
//		   j++;
//		}

		if (obj.terminator == '\n') stream << endl;

//		cin >> pause;
	}
	
	return stream;
}

/*
void DisplayOverlays(void)
{

for(long i=0;i<_len;i++)
{
   for(int j=0;j<MAXLINELENGTH;j++)
   {
	   if(_Strings[i][j])
		   cout << _Strings[i][j];
   }
}




}
*/


// copy constructor
PPbuf::PPbuf(PPbuf &other)
{
	long i=0;
//	int j;

	_len = other._len;
	_Space = other._Space;
	terminator = other.terminator;
 //   _overlay = other._overlay;
	
	// create a new PPbuf	
	_Strings = new String[_len + 1];
	if (!_Strings)
	{
	//	PPerror.StandardError(NOMEM,__FILE__,__LINE__);
		cout << "error";
		exit(0);
	}
	
	if (other._Strings)
	{
		for (i=0; i < _len; i++)
		{
			_Strings[i] = other._Strings[i];
		}
	}
//	strcpy(_Strings[_len],NULL); // = "\0";  // NULL terminate
    _Strings[_len]=BUFEND;  // NULL terminate;


//	_overlay = new (char**[_len + 1]);
//	if (!_overlay)
//	{
	//	PPerror.StandardError(NOMEM,__FILE__,__LINE__);
//		cout << "error";
//		exit(0);
//	}
	
//	if (other._overlay)
//	{
//		for (i=0; i < _len; i++)
//			_overlay[i]=other._overlay[i];
	//		for (j=0;j<MAXLINELENGTH;j++)
	//		   _overlay[i][j] = other._overlay[i][j];
//	}
//	strcpy(_Strings[_len],NULL); // = "\0";  // NULL terminate
//   _overlay[_len]=NULL;  // NULL terminate;

}

// overload the equal operator
PPbuf &PPbuf::operator = (PPbuf &other)
{
	long i=0;
//	int j;
	
	if (this == &other)  // self-assignment
		return *this;
	
	_len = other._len;
	_Space = other._Space;
	terminator = other.terminator;
//	_overlay=other._overlay;

	// if previous delete it
	if (_Strings) delete [] _Strings;
//	if (_overlay) delete [] _overlay;
		
	// create a new PPbuf	
	_Strings = new String[_len + 1];
	if (!_Strings)
	{
//		PPerror.StandardError(NOMEM,__FILE__,__LINE__);
		cout << "error";
		exit(0);
	}

//	_overlay = new (char**[_len + 1]);
//	if (!_overlay)
//	{
//		PPerror.StandardError(NOMEM,__FILE__,__LINE__);
//		cout << "error";
//		exit(0);
//	}

	
	if (other._Strings)
	{
		for (i=0; i < _len; i++)
		{
			_Strings[i] = other._Strings[i];
		}
	}
	//strcpy(_Strings[_len],NULL); // = "\0";  // NULL terminate
	_Strings[_len]=BUFEND;   //NULL terminate

//	if (other._overlay)
//	{
//		for (i=0; i < _len; i++)
		//	for (j=0;j<MAXLINELENGTH;j++)
//			   _overlay[i] = other._overlay[i];
//	}
	//strcpy(_Strings[_len],NULL); // = "\0";  // NULL terminate
//	_overlay[_len]=NULL;   //NULL terminate

	return *this;
}

// overload the conversion operator when using with C routines
PPbuf::operator char **()
{
	long i;
	char **outppbuf = NULL;  //NULL;
	
	outppbuf = new char *[_len + 1];
	if (!outppbuf)
	{
//		PPerror.StandardError(NOMEM,__FILE__,__LINE__);
	    cout << "error";
		delete[] outppbuf;
		exit(0);
	}

//	Soutppbuf = new char *[_len + 1];
//	if (!Soutppbuf)
//	{
//		PPerror.StandardError(NOMEM,__FILE__,__LINE__);
//	    cout << "error";
//		exit(0);
//	}

	for (i=0; i < _len; i++)
	{
		outppbuf[i] = _Strings[i].Strdup();
//		Soutppbuf[i] = _overlay[i].StrDup();
	}
	
	//strcpy(outppbuf[_len],NULL); // = "\0";
	outppbuf[_len]=BUFEND;  // NULL terminate
	return outppbuf;
}

// overload the bracket operator
String &PPbuf::operator[](long index)
{
	assert(index < _len);
	assert(index >= 0);
//	if((index < _len) && (index >= 0))
	    return(_Strings[index]);
//	else
//		return("\0");
}

// overload the bracket operator
String &PPbuf::operator[](int index)
{
	assert(index < _len);
	assert(index >= 0);

//	if((index < _len) && (index >= 0))
	    return(_Strings[index]);
//	else
//		return("\0");
}

istream &operator >> (istream &stream, PPbuf &obj)
{
	long i=0;
	char buf[MAXLINELENGTH];
//	int j;
	
	while (stream)
	{
		if (i == obj._Space)
			obj.GrowBuf();
		stream.getline(buf,MAXLINELENGTH);
		obj._Strings[i] = buf;
//		obj._overlay[i]=NULL;
	//	for(j=0;j<MAXLINELENGTH;j++)
	//	   obj._overlay[i][j] = NULL;
		i++;
	}
	
	obj._len = i;
	//strcpy(obj._Strings[obj._len],NULL);  // = "\0";   // NULL terminate
	obj._Strings[obj._len]=BUFEND;  //BUFEND;  // NULL terminate
	obj.terminator = '\n';
//	obj._overlay[obj._len]=NULL;
	return stream;
}

long PPbuf::size()
{
	return _len;
}

//long PPbuf::Space()
//{
//	return _Space;
//}


void PPbuf::AppendString(String instr)
{
//	cout << "Append string starting" << endl;
//	int j;

	if (_len == _Space)
		this->GrowBuf();
	
	_Strings[_len] = instr;
//	for (j=0;j<MAXLINELENGTH;j++)
//		_overlay[_len] = NULL;
	_len++;

//	cout << "Append string finished" << endl;
}

void PPbuf::AppendString(char *instr)
{
//	cout << "Append string starting" << endl;
//	int j;

	if (_len == _Space)
		this->GrowBuf();
	
	_Strings[_len] = instr;
//	for (j=0;j<MAXLINELENGTH;j++)
//	   _overlay[_len]=NULL;
	_len++;	
//	cout << "Append string finsihing" << endl;
}

// start is the index of the ppbuf to start with
// len is the number of lines to copy
PPbuf PPbuf::dupPPbuf(long start, long len)
{
//	cout << "Starting dup.  constructing PPbuf temp" << endl;
	long i=0;
	int k=0;
	PPbuf temp;
	
	cout << "Starting duplicate assert. len=" << len << ", _len=" << _len << endl;

	assert(len < _len);
	assert(start >= 0);
	
//	cout << "starting dup loop" << endl;
	for (i=start,k=0; i < _len && k < len; i++,k++)
	//	for (j=0;j<MAXLINELENGTH;j++)
		   temp.AppendString(_Strings[i]);
	
//	cout << "finished dup loop.  i="<< i << " j=" << j << endl;
	return temp;
}

String PPbuf::RemoveFirst()
{
	cout << "Starting remove first" << endl;
	String tmp;
	String *pptmp;

	if (!_len)
		return 0;
			
	cout << "Alloacting mem remove first" << endl;
	pptmp = new String[_len];
	if (!pptmp)
	{
//		PPerror.StandardError(NOMEM,__FILE__,__LINE__);
		cout << "error";
		exit(0);
	}
	cout << "mem alloc success" << endl;

	tmp = _Strings[0];
	memcpy((void *)pptmp, (void *)(_Strings+1), (sizeof(String) * _len));
	cout << "mem copy success remove first" << endl;
//	delete _Strings; // do not call the destructor for each object
	delete [] _Strings; // do not call the destructor for each object
	cout << "delete success" << endl;
	_Strings = pptmp;
	cout << "_strings updated - remove first" << endl;
	_len--;
	cout << "remove first finished.  _len=" << _len << endl;
	return tmp;
}

String PPbuf::RemoveLast()
{
	String tmp;
cout << "starting remove last" << endl;
	tmp = _Strings[_len-1];
	cout << "Allocated tmp"<< endl;
	_len--;
cout << "len--.  Finished" << endl;
	return tmp;
}

char PPbuf::GetChar(long line, int column)
{
//cout << "GetChar starting.  Line=" << line << " column=" << column << endl;
	if ((line < _len) && (line >= 0) && (column < MAXLINELENGTH) && (column >= 0))
       return ((int)_Strings[line][column]);
	else
		return(0);
}


String PPbuf::GetString(long line)
{
	if((line < _len) && (line >= 0))
	    return (_Strings[line]);
	else
		return("OUT OF RANGE");
}