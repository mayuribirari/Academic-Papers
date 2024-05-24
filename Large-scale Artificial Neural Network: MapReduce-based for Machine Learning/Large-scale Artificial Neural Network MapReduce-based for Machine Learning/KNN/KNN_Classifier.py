# Imports
from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol, JSONProtocol
from mrjob.step import MRStep
import heapq
import csv
import math

def euclidian_dist(x, y):
    distance = 0.0
    for i in range(len(x)):
        distance += (x[i] - y[i]) ** 2
    return math.sqrt(distance)

class KnnClassifierMapReduce(MRJob):
    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = JSONProtocol
    OUTPUT_PROTOCOL = RawValueProtocol
    def __init__(self, *args, **kwargs):
        super(KnnClassifierMapReduce, self).__init__(*args, **kwargs)
        self.options.points_to_estimate = "C:/Users/mayur/OneDrive/Desktop/CoursesSem2/ECC/project/KNN/input_points.txt"
        self.options.dimensionality = 4
        self.options.knn_type = "classification"
        self.n_neighbours = 5
        with open(self.options.points_to_estimate, "r") as input_file:
            data = list(csv.reader(input_file))
        self.points = {}
        for dp in data:
            self.points[tuple([float(e) for e in dp])] = []


    def mapper_knn(self, _, line):
        data = line.strip().split(",")
        y, features = data[-1], [float(e) for e in data[1:-1]]
        for dp in self.points:
            d_inv = -1 * euclidian_dist(features, dp)
            observation = tuple([d_inv, features, y])
            # if number of nearest neighbours is smaller than threshold add them
            if len(self.points[dp]) < self.n_neighbours:
                self.points[dp].append(observation)
                if len(self.points[dp]) == self.n_neighbours:
                    heapq.heapify(self.points[dp])
            # compare with largest distance and push if it is smaller
            else:
                largest_neg_dist = self.points[dp][0][0]
                if d_inv > largest_neg_dist:
                    heapq.heapreplace(self.points[dp], observation)


    def mapper_knn_final(self):
        yield 1, list(self.points.items())


    def reducer_knn(self, key, points):
        for mapper_neighbors in points:
            merged = None
            mapper_knn = {}
            for k, v in mapper_neighbors:
                mapper_knn[tuple(k)] = v
            if merged is None:
                merged = mapper_knn
            else:
                for point in merged.keys():
                    pq = mapper_knn[point]
                    while pq:
                        if len(merged[point]) < self.n_neighbours:
                            heapq.heappush(merged[point], heapq.heappop(pq))
                        else:
                            largest_neg_dist = merged[point][0][0]
                            if pq[0][0] > largest_neg_dist:
                                heapq.heapreplace(merged[point], heapq.heappop(pq))
        for point in merged.keys():
            estimates = {}
            for neg_dist, features, y in merged[point]:
                estimates[y] = estimates.get(y, 0) + 1
            estimate, counts = max(estimates.items(), key=lambda x: x[-1])
            output = list(point)
            output.append(estimate)
            yield None, ",".join([str(e) for e in output])
         


    def steps(self):     
        return [MRStep(mapper=self.mapper_knn,
                       mapper_final=self.mapper_knn_final,
                       reducer=self.reducer_knn)]
    
if __name__ == "__main__":
    KnnClassifierMapReduce.run()
