package UI;

import Structure.Table;

import java.util.Scanner;

public class ConsoleUserInterface implements UserInterface {

    public static final String RESET = "\u001B[0m";
    public static final String RED = "\u001B[31m";
    public static final String GREEN = "\u001B[32m";
    public static final String YELLOW = "\u001B[33m";
    public static final String BLUE = "\u001B[34m";
    public static final String PURPLE = "\u001B[35m";
    public static final String CYAN = "\u001B[36m";

    private static String getColoredNumber(int number) {
        return switch (number) {
            case 0 -> " " ;
            case 1 -> PURPLE + "I" + RESET;
            case 2 -> GREEN + "F" + RESET;
            case 3 -> YELLOW + "/" + RESET;
            case 4 -> RED + "X" + RESET;
            case 5 -> BLUE + "W" + RESET;
            case -1 -> CYAN + "#" + RESET;
            case -2 -> YELLOW + "#" + RESET;
            default -> PURPLE + number + RESET;
        };
    }

    @Override
    public void display(int[][] table) {
        if (table == null || table.length == 0 || table[0].length == 0) {
            System.out.println("La tabla está vacía o no es válida.");
            return;
        }

        int rows = table.length;
        int cols = table[0].length;

        printBorder(cols);

        for (int i = 0; i < rows; i++) {
            System.out.printf("%2d │", i);
            for (int num : table[i]) {
                System.out.printf(" %s %s│", getColoredNumber(num), RESET);
            }
            System.out.println();
            printBorder(cols);
        }
        printNumbers(cols);
    }

    private static void printBorder(int cols) {
        System.out.print("   ├");
        for (int i = 0; i < cols; i++) {
            System.out.print("───┼");
        }
        System.out.println();
    }

    private static void printNumbers(int cols) {
        System.out.print("    ");
        for (int i = 0; i < cols; i++) {
            System.out.printf(" %2d ", i);
        }
        System.out.println();
    }

    @Override
    public int askForGreaterValueThan(int n, String msg){
        Scanner scanner = new Scanner(System.in);
        int number = 0;

        while (true) {
            System.out.print(msg + ": ");
            if (scanner.hasNextInt()) {
                number = scanner.nextInt();
                if (number > 2) {
                    break; // Exit the loop if the number is valid
                } else {
                    System.out.println("Input inválido: Valor menor o igual a 2");
                }
            } else {
                System.out.println("IInput inválido: No es un número entero");
                scanner.next(); // Clear the invalid input
            }
        }

        return number;
    }

    @Override
    public void askInput(Table tbl) {
        Scanner scanner = new Scanner(System.in);
        boolean ejecutar = false;

        while (!ejecutar) {
            display(tbl.getTable());
            System.out.println("¿Qué deseas hacer?");
            System.out.println("1. Añadir celda peligrosa");
            System.out.println("2. Añadir celda prohibida");
            System.out.println("3. Añadir waypoint");
            System.out.println("4. Eliminar celda");
            System.out.println("5. Ejecutar");
            System.out.print("Selecciona una opción (1-5): ");

            int opcion = scanner.nextInt();

            if (opcion == 5) {
                ejecutar = true;
                break;
            }

            System.out.print("Introduce la fila: ");
            int fila = scanner.nextInt();
            System.out.print("Introduce la columna: ");
            int columna = scanner.nextInt();

            try {
                switch (opcion) {
                    case 1 -> tbl.addDangerousCell(fila, columna);
                    case 2 -> tbl.addProhibitedCell(fila, columna);
                    case 3 -> tbl.addWaypointCell(fila, columna);
                    case 4 -> tbl.removeCell(fila, columna);
                    default -> System.out.println("Opción no válida.");
                }
            } catch (IllegalArgumentException iae) {
                System.out.println(RED + iae.getMessage() + RESET);
            }
        }

        // Preguntar por salida y meta después de ejecutar
        int startRow, startCol, endRow, endCol;
        while (true) {
            System.out.print("Introduce la fila de salida: ");
            startRow = scanner.nextInt();
            System.out.print("Introduce la columna de salida: ");
            startCol = scanner.nextInt();
            if (tbl.isValidCell(startRow, startCol) && tbl.isEmptyCell(startRow, startCol)) break;
            System.out.println("Celda no válida. Inténtalo de nuevo.");
        }
        tbl.addStartCell(startRow, startCol);

        while (true) {
            System.out.print("Introduce la fila de meta: ");
            endRow = scanner.nextInt();
            System.out.print("Introduce la columna de meta: ");
            endCol = scanner.nextInt();
            if (tbl.isValidCell(endRow, endCol) && tbl.isEmptyCell(endRow, endCol)) break;
            System.out.println("Celda no válida. Inténtalo de nuevo.");
        }
        tbl.addFinalCell(endRow, endCol);
    }

    @Override
    public void showPathNotFound() {
        System.out.println(RED + "No existe camino posible para esta disposición. " +
                "Contempla cambiar la posición de las celdas prohibidas si es posible." + RESET);
    }
}