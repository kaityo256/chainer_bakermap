from model import Model

def main():
    unit = 200
    model = Model(unit)
    model.load("baker.model")
    model.export("baker.dat")
    
if __name__ == '__main__':
    main()
