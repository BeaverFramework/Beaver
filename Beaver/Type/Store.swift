/// Responsible of dispatching actions and states.
/// It also holds a state, permitting to notify state changes to its subscribers.
///
/// - Parameters:
///     - ActionType: a type defining the action type
///
/// ## Important Notes: ##
/// 1. Subscribers are store in a set, which means they are called with an indefinite order!
/// 2. Subscribers are blocks, which means that the subscriber itself is responsible
///    of telling if its `self` should be retained or not by the script.
///    In main cases, stages need to declare a `weak self` in order to prevent retain cycles, when the scene
///    needs to declare a `strong self`.
/// 3. Subscribers registration is based on their name. Meaning that two subscribers with
///    the same name would override each others
public final class Store<ActionType: Action> {
    /// State type
    public typealias StateType = State<ActionType.SuccessStateType, ActionType.FailureStateType>

    /// Current state
    fileprivate(set) public var state: StateType

    fileprivate func setState(_ newState: StateType, for action: ActionEnvelop<ActionType>) {
        middleware.run(action, (oldState: state, newState: newState))

        for subscriber in subscribers {
            subscriber.stateDidUpdate(action, state, newState) {
                // do nothing
            }
        }

        self.state = newState
    }

    fileprivate let reducer: Reducer

    fileprivate(set) public var subscribers = Set<Subscriber>()

    fileprivate var middleware: MiddleWare {
        return MiddleWare.composite(middleWares)
    }

    fileprivate var cancellable = Cancellable()

    /// Registered actors
    public let middleWares: [MiddleWare]

    /// Script initialization
    ///
    /// - Parameters:
    ///     - initialState: the first current state
    ///     - director: the director responsible for generating new states
    ///     - actors: a list of actors, responsible of side effects like logging, tracking, ...
    public init(initialState: StateType,
                middleWares: [MiddleWare] = [],
                reducer: @escaping Reducer) {
        self.reducer = reducer
        state = initialState
        self.middleWares = middleWares

        // stateDidSet is not called so we need to dispatch to actors here
        middleware.run(nil, (oldState: nil, newState: initialState))
    }
}

// MARK: - Dispatching

extension Store {
    /// Responsible of dispatching an action and the produced state to subscribers
    public typealias Dispatch = (_ action: ActionEnvelop<ActionType>) -> ()

    /// Dispatching interface
    ///
    /// It is retaining a reference on the script
    public final var dispatch: Dispatch {
        return { (envelop: ActionEnvelop<ActionType>) -> () in
            let cancellable = envelop.action.isLifeCycle ? Cancellable() : self.newCancellable()

            self.middleware.run(envelop, nil)

            self.reducer(envelop, self.state) { newState in
                if !cancellable.isCancelled {
                    self.setState(newState, for: envelop)
                }
            }
        }
    }
}

// MARK: - Subscribing

extension Store {
    /// Adds a new subscriber
    public final func subscribe(_ subscriber: Subscriber) {
        subscribers.insert(subscriber)

        // Dispatching the state update permits to avoid infinite recursions when
        // the `stateDidUpdate` method implementation refers the script
        DispatchQueue.main.async {
            subscriber.stateDidUpdate(nil, nil, self.state) {
                // do nothing
            }
        }
    }

    /// Build and add a new subscriber
    ///
    /// - Parameters:
    ///     - name: the subscriber's name
    ///     - stateDidUpdate: the subscriber state update handler
    public final func subscribe(name: String, stateDidUpdate: @escaping Subscriber.StateDidUpdate) {
        subscribe(Subscriber(name: name, stateDidUpdate: stateDidUpdate))
    }

    /// Removes a subscriber
    public final func unsubscribe(_ subscriber: Subscriber) {
        subscribers.remove(subscriber)
    }
}

// MARK: - Description

extension Store: CustomDebugStringConvertible {
    public var debugDescription: String {
        return "\(self) - subscribers: [\(subscribers.map { $0.debugDescription })]"
    }
}

// MARK: - Cancellation

fileprivate extension Store {
    final class Cancellable {
        private(set) var isCancelled: Bool = false

        func cancel() {
            isCancelled = true
        }
    }

    func newCancellable() -> Cancellable {
        cancellable.cancel()
        let actionCancellable = Cancellable()
        cancellable = actionCancellable
        return actionCancellable
    }
}