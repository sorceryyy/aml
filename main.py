from dataProcessor import DataProcessor

if __name__ == "__main__":
    data = DataProcessor()
    pdr = data.get_pdr()
    ans =pdr.pdr_position()
    print(ans)