package Structure;

import java.util.ArrayList;
import java.util.List;

public class OrderedNodeList {

    List<Node> nodes = new ArrayList<>();

    public void insert(Node node) {
        if (nodes.contains(node))
            return;

        if (nodes.isEmpty() || nodes.getFirst().getHeuristic() > node.getHeuristic()) {
            // If the new heuristic is smaller than the first, add at the beginning
            nodes.addFirst(node);
            return;
        }

        if (nodes.getLast().getHeuristic() < node.getHeuristic()) {
            // If the new heuristic is bigger than the last, add at the end
            nodes.addLast(node);
            return;
        }

        // Binary search
        int left = 0;
        int right = nodes.size() - 1;

        // Goes narrowing it down
        while (left <= right) {
            int mid = left + (right - left) / 2;
            double midHeuristic = nodes.get(mid).getHeuristic();

            if (midHeuristic == node.getHeuristic()) {
                nodes.add(mid + 1, node);
                return;
            } else if (midHeuristic < node.getHeuristic()) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        // Insert in the correct position
        nodes.add(left, node);
    }

    public boolean isEmpty() {
        return nodes.isEmpty();
    }

    public Node poll() {
        return nodes.removeFirst();
    }

    public void clear(){
        nodes.clear();
    }

    public void addAll(List<Node> children) {
        for (Node child : children) {
            insert(child);
        }
    }
}
