#include <iostream>
using namespace std;
class Subject{
	public:
		Subject()
		{
			observers = new vector<Observer*>();
		}
		
		virtual ~Subject()
		{
			
		}
		
		virtual void attach(Observer* obs)
		{
			if(obs != NULL)
			{
				observers.push_back(obs);
			}
		}
		
		virtual void detach(Observer* obs)
		{
			if(obs != NULL)
			{
				observers.remove(obs);
			}
		}
		
		virtual void notify()
		{
			for(int observers_i = 0; observers_i < observers.size(); ++observers_i++)
			{
				observers[observers_i]->update();
			}
		}
	private:
		vector<Observer*>* observers;
}; 

class ConcreteSubject: Subject{
	public:
		ConcreteSubject();
		~ConcreteSubject();
};

class Observer{
	public:
		Observer();
		virtual ~Observer();
		virtual update() = 0;
};

class ConcreteObserverA

int main()
{
	
	return 0;
}
